"""训练辅助工具：设备初始化、学习率、随机种子、模型加载与断点恢复。

文件末尾的奖励模型、数学求值和重复惩罚供后续训练阶段使用，预训练入口不会调用。
"""
import ast
import math
import operator
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel
from minimind.model.model_minimind import MiniMindForCausalLM

# 默认资源目录以源码位置为基准，使 uv 命令和 python -m 使用相同的文件位置。
PACKAGE_DIR = Path(__file__).resolve().parents[1]

# ======== 模型参数量 ========
def get_model_params(model, config):
    """打印参数总量，以及 MoE 每个 token 大致激活的参数量，单位为百万。

    普通模型使用全部参数；MoE 按配置估算被选中的专家参数量，因此总参数量
    和激活参数量可能不同。这里用于展示，不改变参数的 requires_grad，也不返回统计值。
    """
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


# ======== 主进程判断 ========
def is_main_process():
    """返回是否为主进程：非分布式运行，或分布式 rank 为 0。

    用于让多个训练进程共享一份日志和检查点，避免重复打印与同时写同一文件。
    """
    return not dist.is_initialized() or dist.get_rank() == 0


# ======== 日志 ========
def Logger(content):
    """仅在主进程打印 content；保留多 GPU 训练时的一份终端日志。"""
    if is_main_process():
        print(content)


# ======== 混合精度类型 ========
def resolve_dtype(dtype, device):
    """返回训练精度名称，在 CUDA 不原生支持 bfloat16 时回退到 float16。

    该函数不转换模型或张量；实际 autocast 上下文与 GradScaler 由训练入口创建。
    其他精度名称和非 CUDA 设备会直接沿用传入值。
    """
    if dtype != "bfloat16" or not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return dtype
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return dtype
    Logger(f"{torch.cuda.get_device_name(device)} 不原生支持 bfloat16，已回退为 float16")
    return "float16"


# ======== 学习率 ========
def get_lr(current_step, total_steps, lr, warmup_ratio=0.0):
    """根据全局 batch 进度返回学习率：可选线性预热，随后余弦衰减。

    lr 是峰值学习率；warmup_ratio 决定预热长度。预热后学习率逐渐下降，
    在 total_steps 时达到 lr 的 10%。本项目按 batch 调用，梯度累积时
    current_step 仍逐 batch 增加，不等于优化器已经更新的次数。
    """
    warmup = max(1, int(warmup_ratio * total_steps)) if warmup_ratio > 0 else 0
    if current_step < warmup:
        return lr * (current_step + 1) / warmup
    angle = math.pi * (current_step - warmup) / max(1, total_steps - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(angle)))


# ======== 分布式初始化 ========
def init_distributed_mode():
    """读取 torchrun 的环境变量，初始化 NCCL 通信并绑定当前进程的 GPU。

    RANK 表示全局进程编号，LOCAL_RANK 表示本机 GPU 编号。返回 LOCAL_RANK；
    没有分布式环境变量时不创建进程组，返回 0，后续按单进程运行。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# ======== 随机种子 ========
def setup_seed(seed: int):
    """统一设置 Python、NumPy、PyTorch CPU/CUDA 随机种子，并配置 cuDNN。

    用于尽量复现模型初始化、打乱顺序和 dropout 等随机操作；不同设备、
    软件版本或非确定性算子之间仍不保证结果逐位一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======== 断点保存与续训 ========
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir=PACKAGE_DIR / 'checkpoints', **kwargs):
    """保存或读取训练断点；是否传入 model 决定工作模式。

    保存时需传入 optimizer：导出半精度模型权重，同时保存包含原精度权重、
    优化器、从 0 开始的 epoch、已完成 batch 数 step 等信息的续训文件。
    kwargs 可携带 scaler 等有 state_dict 的对象，或其他需要保存的普通值。
    先写临时文件，再原子替换目标文件，减少读到半份检查点的风险。

    不传 model 时读取续训文件并返回字典，不存在则返回 None；GPU 数变化时
    按 world_size 比例折算 step。这是批次进度换算，不保证与原训练逐步等价。
    当前未保存随机数状态或尚未更新的 .grad，保存点应与梯度累积周期对齐。
    默认目录为包内 checkpoints；显式传入的相对路径按当前工作目录解析。
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


# ======== 模型初始化 ========
def init_model(lm_config, from_weight='pretrain', tokenizer_path=PACKAGE_DIR / 'model', save_dir=PACKAGE_DIR / 'out', device='cuda'):
    """返回放到 device 的模型和配套分词器；可从已有权重初始化。

    lm_config 定义模型结构；from_weight='none' 时随机初始化，否则从
    save_dir 加载对应前缀的权重。这里不恢复优化器和训练进度，完整续训由
    lm_checkpoint 配合训练入口完成。默认目录分别为包内 model、out；显式传入
    的相对路径按当前工作目录解析。分词器词表及特殊 ID 需与模型配置匹配。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


# ======== 跳过批次采样器 ========
class SkipBatchSampler(Sampler):
    """把样本索引组成 batch，并在续训时跳过已处理的批次。"""

    def __init__(self, sampler, batch_size, skip_batches=0):
        """接收可迭代且有长度的索引源、单批样本数，以及要跳过的 batch 数。

        sampler 可以是 DistributedSampler，也可以是已经打乱的索引列表；
        skip_batches 的单位是批次而不是样本数。
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """逐批产生样本索引列表；先跳过已完成批次，并保留末尾不足一批的样本。

        DataLoader 根据这些索引调用数据集的 __getitem__，随后堆叠输入张量。
        """
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """返回跳过后的 batch 数；向上取整以包含最后不足 batch_size 的一批。"""
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


# ======== 奖励模型 ========
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        """加载提供 get_score 接口的奖励模型和分词器，置于推理模式。

        用于后续强化学习阶段，不参与当前预训练；trust_remote_code=True
        允许执行模型仓库提供的自定义实现，因此 model_path 应指向可信来源。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        """把对话历史和候选回答交给奖励模型，返回截断到 [-3, 3] 的分数。

        messages 是含 role、content 的消息列表，最后一条作为当前问题，
        response 是待评分的回答；关闭梯度可减少评分过程的计算和显存开销。
        """
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)


# ======== 数学表达式安全求值 ========
def safe_math_eval(expression):
    """解析并计算有限语法的数学表达式，用于后续数学奖励等辅助逻辑。

    使用 AST 白名单遍历，不使用 eval；支持基本算术、指定 math 函数和
    pi/e/tau，返回计算结果。对输入长度、幂运算规模作限制；不支持的语法
    抛出 ValueError，除零等数值错误仍由相应运算抛出。
    """
    def pow_guard(base, exp):
        """限制指数大小和结果量级，再计算 base ** exp，避免过大的幂运算。"""
        if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * math.log10(abs(base)) > 100): raise ValueError('幂运算结果过大')
        return base ** exp
    def resolve(node):
        """提取 name 或 math.name 中的名称，供后续白名单查找；其余返回 None。"""
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute) and getattr(node.value, 'id', '') == 'math': return node.attr
    def walk(node):
        """递归求值数字、常量、运算符及白名单函数调用，拒绝其他 AST 节点。"""
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        name = resolve(node)
        if name in consts: return consts[name]
        if isinstance(node, ast.UnaryOp): return unary_ops[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp): return bin_ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call) and not node.keywords: return funcs[resolve(node.func)](*map(walk, node.args))
        raise ValueError(f'不支持的表达式语法: {type(node).__name__}')
    consts = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
    funcs = {'pow': pow_guard, **{n: getattr(math, n) for n in 'sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil trunc fabs fmod hypot gcd degrees radians'.split()}}
    bin_ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: pow_guard}
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    chars = str.maketrans({'^': '**', '×': '*', '÷': '/', '−': '-', '²': '**2', '³': '**3', '（': '(', '）': ')'})
    expr = str(expression).translate(chars).strip()
    if not expr or len(expr) > 512: raise ValueError('表达式为空或过长')
    try:
        return walk(ast.parse(expr, mode='eval').body)
    except (KeyError, SyntaxError, TypeError):
        raise ValueError('不支持的表达式语法') from None


# ======== 重复惩罚 ========
def rep_penalty(text, n=3, cap=0.5):
    """按重复 n-gram 比例返回 [0, cap] 范围的惩罚值，重复越多惩罚越大。

    中日韩字符按单字符切分，其他文本按单词或符号切分；样本不足 n 个
    单位时返回 0。用于后续生成文本评分，不改变模型内部的 token 采样分数。
    """
    cjk = ("\u3007\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
           "\U00020000-\U0002ee5f\U0002f800-\U0002fa1f\U00030000-\U0003347f")
    toks = re.findall(rf"[{cjk}]|[^\W{cjk}]+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0
