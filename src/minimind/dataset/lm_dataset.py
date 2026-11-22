"""把 JSONL 中的原始文本转换为定长的预训练输入和监督标签。"""

import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

# DataLoader 可以启动多个进程；关闭分词器内部并行，避免重复创建线程。
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    """每条样本读取 text 字段；完整文本中的有效 token 都参与下一词预测。"""

    def __init__(self, data_path, tokenizer, max_length=512):
        """加载 JSON/JSONL 数据，保存分词器和包含 BOS、EOS、PAD 的总序列长度。

        data_path 指向带有 text 字段的数据文件。load_dataset 返回支持按索引
        读取的 Hugging Face Dataset，并按其缓存配置保存解析后的数据。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        """返回样本条数，供采样器和 DataLoader 计算批次数。"""
        return len(self.samples)

    def __getitem__(self, index):
        """返回两个形状均为 [max_length]、类型为 long 的张量。

        input_ids 由 BOS、截断后的正文、EOS 和右侧 PAD 组成；labels 复制
        input_ids，再把 PAD 位置改成 -100，使交叉熵忽略这些补齐位置。
        标签在这里不移动；模型 forward 内部会对 logits 和 labels 错开一位。
        """
        sample = self.samples[index]
        # 为文本开始、结束标记各预留一个位置，且由本数据集手动添加标记。
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # 这要求 PAD 与 EOS 使用不同 ID，否则结束标记也会被忽略。
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
