"""预训练入口：初始化环境，再循环执行前向计算、反向传播、参数更新和保存。"""

import argparse
import os
import time
import warnings
from contextlib import nullcontext

# 此导入用于先加载 pyarrow，规避 Windows 下与 torch 的 DLL 冲突，不能按未使用删除。
import datasets  # noqa: F401
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from minimind.dataset.lm_dataset import PretrainDataset
from minimind.model.model_minimind import MiniMindConfig
from minimind.trainer.trainer_utils import (
    PACKAGE_DIR,
    TARGET_DIR,
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    resolve_dtype,
    setup_seed,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """训练一轮，并在指定间隔输出当前 batch 的损失、保存模型和续训状态。

    epoch 是从 0 开始的轮次；loader 提供 input_ids、labels；iters 是本轮
    总 batch 数，包含续训时跳过的批次；start_step 是已经完成的批次数。
    wandb 可传入日志模块，本入口实际使用 swanlab。

    模型、优化器、GradScaler、混合精度上下文和参数来自入口初始化的全局变量。
    step 每处理一个 batch 加一，累积若干 batch 的梯度后才更新一次参数，
    因而日志里的 step 数与 optimizer.step 的次数不同；本函数无返回值。
    """
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 用跨轮次的 batch 进度调度学习率；同一累积周期内也会更新这个值。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.warmup_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # 普通模型的 aux_loss 为 0；MoE 模型还需加入专家负载均衡损失。
            loss = res.loss + res.aux_loss
            # 固定除以累积步数，避免完整累积周期的梯度随 batch 数成倍放大。
            # 当前实现对轮末不足一个累积周期的批次也使用此分母，权重会偏小。
            loss = loss / args.accumulation_steps

        # float16 时缩放损失以减轻梯度下溢；禁用 scaler 时等价于普通 backward。
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            # 必须先还原梯度，再裁剪范数，否则裁剪的是放大后的梯度。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # 梯度默认会相加；参数更新后清空，开始下一个累积周期。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 还原用于日志的 batch 损失；它不是整个累积周期或整轮的平均损失。
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 单独导出半精度权重用于加载；完整续训状态由 lm_checkpoint 保存。
            # 续训状态不含尚未更新的梯度，周期保存间隔应与梯度累积周期对齐。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb)
            model.train()
            del state_dict

        del input_ids, labels, res, loss


def main():
    """解析命令行并初始化训练，供 uv run pretrain 和 python -m 共同调用。

    初始化本模块 train_epoch 使用的模型、优化器等全局对象，再按设定轮数训练。
    默认数据和分词器路径基于源码位置，产物保存在项目根目录的 target 下；用户显式
    指定的相对路径基于启动时的工作目录。
    仅导入模块不会开始训练，--help 会在初始化模型和创建输出目录之前退出。
    """
    global args, lm_config, model, optimizer, scaler, autocast_ctx

    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default=str(TARGET_DIR / 'out'), help="模型保存目录（默认项目根目录下的 target/out）")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="线性warmup占总步数比例，0为关闭")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0 if os.name == 'nt' else 8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default=str(PACKAGE_DIR / 'dataset' / 'pretrain_t2t_mini.jsonl'), help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight) if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # autocast 主要降低中间计算的精度；这里没有直接把模型参数整体转为半精度。
    device_type = "cuda" if torch.device(args.device).type == "cuda" else "cpu"
    args.dtype = resolve_dtype(args.dtype, args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.save_dir, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # bfloat16 的数值范围较大，因此这里只为 float16 启用梯度缩放。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    # 续训需要恢复 AdamW 的动量等状态，仅加载模型权重不等于恢复训练进度。
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        if ckp_data.get('scaler'): scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 每轮重新打乱样本；各 DDP 进程通过采样器获取自己的样本分片。
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
