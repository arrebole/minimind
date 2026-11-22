"""终端模型测试：预训练权重默认原文续写，SFT 权重可选择聊天模板。"""

import argparse
import math
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer, TextStreamer
from transformers.utils import logging as transformers_logging

from minimind.model.model_minimind import MiniMindConfig, MiniMindForCausalLM


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parents[1]


def parse_args():
    """读取权重、设备和生成参数；提前拒绝无效长度与采样范围。"""
    parser = argparse.ArgumentParser(description="MiniMind 终端对话与文本续写测试")
    parser.add_argument('--weight_path', type=Path, default=PROJECT_ROOT / 'target/out/pretrain_768.pth', help="模型权重或完整续训文件")
    parser.add_argument('--tokenizer_path', type=Path, default=PACKAGE_DIR / 'model', help="分词器目录")
    parser.add_argument('--device', default='auto', help="auto、cpu 或 cuda:0 等 CUDA 设备")
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto', help="计算精度；CPU 使用 float32")
    parser.add_argument('--prompt', help="只生成一次；不传则进入交互模式")
    parser.add_argument('--max_new_tokens', type=int, default=256, help="每次最多生成的 token 数")
    parser.add_argument('--max_input_tokens', type=int, default=1024, help="输入 token 上限，包含可选聊天历史")
    parser.add_argument('--temperature', type=float, default=0.8, help="采样温度，越低通常越集中")
    parser.add_argument('--top_p', type=float, default=0.9, help="累计概率采样阈值，范围 (0, 1]")
    parser.add_argument('--top_k', type=int, default=50, help="保留最高分的 k 个候选，0 表示关闭")
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help="已生成 token 的重复惩罚，1 表示关闭")
    parser.add_argument('--greedy', action='store_true', help="使用贪心解码，关闭随机采样")
    parser.add_argument('--seed', type=int, default=42, help="控制随机采样的种子")
    parser.add_argument('--chat_template', action='store_true', help="启用分词器聊天模板和多轮历史，供 SFT 模型使用")
    parser.add_argument('--no_stream', action='store_true', help="生成完成后一次性打印正文")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0:
        parser.error('max_new_tokens 和 max_input_tokens 必须大于 0')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('temperature 必须是大于 0 的有限数')
    if not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error('top_p 必须在 (0, 1] 内，top_k 不能小于 0')
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0:
        parser.error('repetition_penalty 必须是大于 0 的有限数')
    if args.prompt is not None and not args.prompt.strip():
        parser.error('prompt 不能为空')
    return args


def resolve_device_dtype(args):
    """选择设备和计算精度：CUDA 优先 BF16，CPU 使用 FP32；不修改模型参数。"""
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if device.type not in {'cuda', 'cpu'}:
        raise ValueError('当前测试脚本支持 cpu 或 cuda 设备')
    if device.type == 'cpu':
        if args.dtype not in {'auto', 'float32'}:
            raise ValueError('CPU 测试请使用 --dtype auto 或 --dtype float32')
        return device, torch.float32
    if not torch.cuda.is_available():
        raise ValueError('CUDA 不可用，请检查 GPU 环境或改用 --device cpu')
    with torch.cuda.device(device):
        bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
    dtype_name = args.dtype
    if dtype_name == 'auto':
        dtype_name = 'bfloat16' if bf16_supported else 'float16'
    if dtype_name == 'bfloat16' and not bf16_supported:
        raise ValueError('当前 GPU 不原生支持 BF16，请使用 --dtype auto 或 float16')
    return device, getattr(torch, dtype_name)


def load_model(args):
    """严格加载权重，返回模型、分词器、设备和计算精度，不恢复训练优化器。

    权重可以是导出的 state_dict，或包含 model 字段的续训文件。隐藏维度、
    层数、词表、注意力头与专家结构从张量形状恢复；RoPE 等无权重配置沿用
    当前训练入口的 MiniMind 默认值。严格校验键名和形状，避免漏加载后继续推理。
    """
    path = args.weight_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'找不到模型权重：{path}')
    with path.open('rb') as stream:
        if stream.read(64).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('权重仍是 Git LFS 指针，请先运行 git lfs pull')
    weights = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if isinstance(weights, dict) and isinstance(weights.get('model'), dict):
        weights = weights['model']
    vocab_size, hidden_size = weights['model.embed_tokens.weight'].shape
    layer_ids = [int(name.split('.')[2]) for name in weights if name.startswith('model.layers.')]
    first_layer = 'model.layers.0.'
    head_dim = weights[first_layer + 'self_attn.q_norm.weight'].numel()
    use_moe = first_layer + 'mlp.gate.weight' in weights
    mlp_key = 'mlp.experts.0.gate_proj.weight' if use_moe else 'mlp.gate_proj.weight'
    intermediate_size = weights[first_layer + mlp_key].shape[0]
    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=max(layer_ids) + 1,
        use_moe=use_moe,
        vocab_size=vocab_size,
        head_dim=head_dim,
        num_attention_heads=weights[first_layer + 'self_attn.q_proj.weight'].shape[0] // head_dim,
        num_key_value_heads=weights[first_layer + 'self_attn.k_proj.weight'].shape[0] // head_dim,
        intermediate_size=intermediate_size,
        moe_intermediate_size=intermediate_size,
        num_experts=weights[first_layer + 'mlp.gate.weight'].shape[0] if use_moe else 4,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path.expanduser().resolve(), local_files_only=True)
    if len(tokenizer) != vocab_size or tokenizer.bos_token_id != config.bos_token_id or tokenizer.eos_token_id != config.eos_token_id:
        raise ValueError('分词器词表或 BOS/EOS ID 与模型配置不匹配')
    model = MiniMindForCausalLM(config)
    model.load_state_dict(weights, strict=True)
    device, dtype = resolve_device_dtype(args)
    # 保留 FP32 参数及 RoPE 表，生成时用 autocast 降低计算精度，与训练入口一致。
    return model.to(device).eval(), tokenizer, device, dtype


def encode_prompt(prompt, history, tokenizer, model, args, device):
    """构造 [1, 输入长度] 的 token ID；默认使用训练时的 BOS + 原文格式。

    仅启用 chat_template 时才加入角色和多轮历史。输入超限时明确报错，避免
    静默截断聊天模板或超出 RoPE 位置范围；历史过长可用 /clear 清空。
    """
    if args.chat_template:
        text = tokenizer.apply_chat_template(history + [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
    else:
        token_ids = [tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False).input_ids
    limit = min(args.max_input_tokens, model.config.max_position_embeddings - args.max_new_tokens)
    if limit <= 0:
        raise ValueError('生成长度超过模型上下文容量，请减小 max_new_tokens')
    if len(token_ids) > limit:
        raise ValueError(f'输入有 {len(token_ids)} 个 token，上限为 {limit}；请缩短输入或使用 /clear')
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def generate_reply(prompt, history, model, tokenizer, device, dtype, args):
    """复用 KV 缓存生成一段文本，只打印并返回新增正文，不把提示词重复输出。

    TextStreamer 将逐 token 输出解码为终端文本；no_stream 时生成完再打印。
    logits_to_keep=1 仅计算最后位置的词表投影，推理过程不创建梯度。
    """
    input_ids = encode_prompt(prompt, history, tokenizer, model, args, device)
    streamer = None if args.no_stream else TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    context = torch.autocast('cuda', dtype=dtype) if device.type == 'cuda' and dtype != torch.float32 else nullcontext()
    print('模型：', end='', flush=True)
    with torch.inference_mode(), context:
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=min(args.top_k, model.config.vocab_size),
            eos_token_id=tokenizer.eos_token_id,
            do_sample=not args.greedy,
            repetition_penalty=args.repetition_penalty,
            use_cache=True,
            streamer=streamer,
            logits_to_keep=1,
        )
    reply = tokenizer.decode(output[0, input_ids.shape[1]:].tolist(), skip_special_tokens=True)
    if args.no_stream:
        print(reply)
    if not reply.strip():
        print('（模型未生成可见正文，可能直接输出了结束符。）')
    return reply


def main():
    """加载模型后进行一次生成，或进入支持 /clear、/exit 的终端交互循环。"""
    args = parse_args()
    # 不在对话正文中穿插库接口的弃用提示；实际加载和生成错误仍会报告。
    transformers_logging.set_verbosity_error()
    warnings.filterwarnings('ignore', message=r'`rope_scaling` is deprecated.*', category=FutureWarning)
    torch.manual_seed(args.seed)
    try:
        model, tokenizer, device, dtype = load_model(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f'加载失败：{error}') from error
    print(f'已加载：{args.weight_path.resolve()}\n设备：{device}，计算精度：{dtype}')
    if args.chat_template:
        print('聊天模板模式：保留多轮历史，适用于经过 SFT 的模型。')
    else:
        print('预训练模式：每轮独立进行原文续写，不使用角色模板。指令和多轮能力需后续 SFT。')
    if args.prompt is not None:
        try:
            generate_reply(args.prompt, [], model, tokenizer, device, dtype, args)
        except ValueError as error:
            raise SystemExit(f'生成失败：{error}') from error
        return
    print('输入文本开始测试；/clear 清空历史，/exit 退出；Ctrl+C 或 Ctrl+D 也可退出。')
    history = []
    while True:
        try:
            prompt = input('\n你：').strip()
            if prompt.lower() in {'/exit', 'exit', 'quit'}:
                break
            if prompt == '/clear':
                history.clear()
                print('已清空历史。')
                continue
            if not prompt:
                continue
            reply = generate_reply(prompt, history, model, tokenizer, device, dtype, args)
            if args.chat_template:
                history.extend([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': reply}])
        except ValueError as error:
            print(f'输入错误：{error}')
        except (EOFError, KeyboardInterrupt):
            print('\n已退出。')
            break


if __name__ == '__main__':
    main()
