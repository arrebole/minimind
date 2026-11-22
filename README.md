## Minimind

### 启动预训练

将 `pretrain_t2t_mini.jsonl` 放入 `src/minimind/dataset/`，分词器文件放入
`src/minimind/model/`。在项目根目录执行：

```bash
uv run pretrain
```

`uv` 会自动安装项目及锁定依赖，再调用 `pyproject.toml` 中配置的
`minimind.trainer.train_pretrain:main`。

```bash
uv run pretrain --data_path /数据的绝对路径/pretrain_t2t_mini.jsonl
uv run pretrain --help
```

默认分词器和数据路径以源码包的位置为基准，训练产物统一保存在项目根目录的 `target/` 下：

- `target/out/`：导出的模型权重，`--from_weight` 默认也从这里加载。
- `target/checkpoints/`：断点权重及包含优化器、训练进度等信息的续训文件，
  `--from_resume 1` 默认从这里恢复。

使用 `--save_dir` 可以自定义导出权重及 `--from_weight` 的加载目录；续训目录仍为
项目根目录下的 `target/checkpoints/`。显式指定的相对路径按启动目录解析。
也可以使用同一个入口执行 `uv run python -m minimind.trainer.train_pretrain`。

### 对话与续写测试

默认加载 `target/out/pretrain_768.pth`，自动识别模型结构并优先使用 CUDA：

```bash
uv run chat
uv run chat --prompt "人工智能是" --max_new_tokens 128
uv run chat --weight_path target/checkpoints/pretrain_768_resume.pth
```

交互模式输入 `/exit` 退出、`/clear` 清空历史；默认流式输出，使用 `--no_stream`
可以等待生成完成后一次性打印。`--device cpu` 可在 CPU 上测试。
预训练权重默认采用 BOS + 原文的独立续写格式，没有聊天角色和历史；完成 SFT 后，
可用 `--chat_template` 启用聊天模板与多轮历史。`--greedy` 可关闭随机采样。

### 使用 Git LFS 记录训练产物

`target/` 下的完整文件由 `.gitattributes` 配置为 Git LFS 跟踪，保存过程中的
`.tmp` 和下载过程中的 `.part` 文件忽略。使用前安装 Git LFS，并在仓库执行：

```bash
git lfs install --local
git add .gitattributes target/
git lfs ls-files
```

暂存后，Git 记录小型指针，完整模型内容保存在本地 LFS 对象库；后续 `git push`
时由 Git LFS 上传模型内容。自定义到 `target/` 以外的输出目录不受此跟踪规则覆盖。

### minimind-3 架构

![LLM-structure-moe](docs/LLM-structure.jpg)
![LLM-structure-moe](docs/LLM-structure-moe.jpg)
