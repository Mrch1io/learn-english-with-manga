# 使用指南

[返回项目首页](README.md)

支持日文漫画翻译为英文、人工校对和日语点读。以下步骤适用于 Windows + NVIDIA CUDA，所有命令在项目根目录运行；示例输入路径请替换为自己的文件。

## 安装

使用 Python 3.12 创建环境：

```powershell
py -3.12 -m venv .venv
```

直接安装 [requirements.txt](requirements.txt) 中的项目依赖，其中已固定 CUDA 12.6 版 PyTorch `2.10.0+cu126` 和 torchvision `0.25.0+cu126`，无需另行安装 PyTorch。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'"
```

气泡检测使用项目根目录的 `best.pt`。其他模型与运行时按下表准备，下载来源、固定版本和许可见[第三方说明](THIRD_PARTY.md)。

| 资源 | 位置与用途 |
|---|---|
| VNTL GGUF | `runtime/vntl/vntl-llama3-8b-v2-hf.Q4_K_M.gguf`，默认翻译模型 |
| CUDA llama.cpp b11706 | `runtime/llama.cpp/b11706/llama-server.exe`，由程序管理翻译服务 |
| Manga-OCR、LaMa | 首次需要时下载并缓存；分别用于日文识别、复杂背景清字 |
| Magi v2 | `runtime/magiv2`，默认用于气泡外文字检测；可用 `--bubbles-only` 关闭 |
| VOICEVOX | 可选，仅日语点读需要，见[声线设置](#日语点读与声线) |

默认检测流程还需安装 Magi 依赖并下载权重：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-magi.txt
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('ragavsachdeva/magiv2', revision='fbc890fec52977142e8ee00bfe26e9458b65517c', local_dir='runtime/magiv2')"
```

Magi 仅限学术研究用途。选择可选的 Qwen 后端还需 `Qwen/Qwen2.5-3B-Instruct` 基座及根目录的 `manga-lora-adapter/`；默认 VNTL 不需要它们。

## 启动与翻译

```powershell
.\launch.bat
```

启动后进入阅读器并打开随仓库提供的三页样例。样例英文和角色经过人工整理，用于展示操作；查看样例不加载推理模型，播放需要 VOICEVOX。

- **逐页翻译**：通过 **Import → Images / JSON... / Folder...** 导入，在 **Tools → Processing** 设置作品名与输出目录，点击 **Tools → Translate Current Page** 生成译图。
- **批量翻译**：进入 **Tools → Tasks**，选择图片或文件夹、输出目录和作品名，点击 **Start Translation**。勾选 **Text Only** 可先保存识别与翻译文本。
- **阅读结果**：用 **Show Original / Show Translation** 切换图片。已有纯文本结果需通过 **Tools → Export Saved Translation** 生成译图。

启动器优先使用 `.venv\Scripts\python.exe`，否则读取 `.gpu-python` 中的解释器路径；两种启动方式均要求 CUDA 可用。

## 命令行

```powershell
# 递归处理文件夹，页面按自然顺序排列
.\run.bat ".\input" -o ".\translated_output" --work "Book1"

# 先保存文本，稍后在阅读器中校对或导出
.\run.bat ".\input" -o ".\text_output" --work "Book1" --text-only
.\run.bat --read ".\text_output"

# 不使用 Magi，只处理气泡内文字
.\run.bat ".\input\page.jpg" -o ".\translated_output" --work "Book1" --bubbles-only

# 查看全部参数
.\.venv\Scripts\python.exe main.py --help
```

翻译必须提供输入路径。同一命令可续跑；人工修订默认保留，配置变化时标记待复核。`--force` 会允许重新生成并覆盖人工修订。请保留原图，以及结果中配套的 JSON、PNG 和 `.clean.png`；术语与角色信息另存于 `data/memory.sqlite3`。

常用选项：

| 参数 | 用途 |
|---|---|
| `--offline` | 仅使用已有权重和缓存，需先完成下载 |
| `--device cuda:1` | 指定 GPU，默认 `cuda:0`；不支持 CPU 推理 |
| `--memory-policy staged` | 默认分阶段管理显存；`resident` 用于显存充足的环境 |
| `--translator vntl` | 默认后端；`qwen` 使用另行准备的基座与 LoRA |
| `--gguf` / `--llama-server` | 指定翻译权重或 llama.cpp 可执行文件 |
| `--mode individual` | 逐框翻译；默认自动选择 VNTL 分组、Qwen 逐框 |
| `--no-speaker-context` | 关闭已确认角色信息的翻译上下文 |
| `--font` | 指定支持所需字符的 TTF、OTF 或 TTC 字体 |

## 校对与术语

阅读器顶部 **Workbench** 进入校对。**Save Translation & Redraw** 修改英文；**Save Source & Retranslate** 修改日文并重译当前框。纯文本结果编辑后仍需 **Export Image** 导出译图。

局部 OCR 使用 **Enable Local OCR → Preview OCR Candidate → Apply Candidate & Retranslate**，预览不会修改原文。补清残字使用 **Tools → Image Cleanup → Clean Selection & Export**。**Tools → Revision Manager** 可查看保留的修订并跳转校对。

术语按作品隔离；锁定译名可用于保持名称一致：

```powershell
.\run.bat --work "Book1" --set-term "高梨美咲" "Misaki Takanashi" --lock
.\run.bat --work "Book1" --list-terms
.\run.bat --work "Book1" --export-terms ".\terms.json"
.\run.bat --work "Book1" --import-terms ".\terms.json"
```

## 日语点读与声线

从 [VOICEVOX Engine 0.25.2 发布页](https://github.com/VOICEVOX/voicevox_engine/releases/tag/0.25.2)下载 Windows DirectML 版本，解压至 `runtime/voicevox/windows-directml/`，确认其中有 `run.exe`。

阅读器 **Settings → Start / Connect** 连接或启动引擎，**Refresh Voices** 更新声音列表，**Preview Voice** 试听。点击气泡朗读保存的日文；可在设置中关闭 **Play on bubble click**。侧栏 **Bind Selected Bubble** 确认角色，**Settings → Manage Character → Save Name & Default Voice** 保存名称和默认声线。Magi 的角色建议需人工确认。

引擎默认监听 `127.0.0.1:50021`，关闭阅读器后仍独立运行，日志位于 `data/voicevox/`。CLI 需先启动引擎，再用 `--list-voices` 查询声音；`--speak-page` 仅生成 WAV。播放功能目前适配 Windows，声音使用及署名要求见[第三方说明](THIRD_PARTY.md)。

## 训练气泡检测器

使用 [tools/train.py](tools/train.py)，先自行准备 Ultralytics 格式的数据集及 `data.yaml`：

```powershell
.\.venv\Scripts\python.exe tools/train.py --data ".\dataset\data.yaml" --weights yolov8m.pt --epochs 30
```

首次训练默认保存至 `output/training/bubble-detector/`，重复运行时目录名自动递增。将实际输出中的 `weights/best.pt` 放入项目根目录即可用于检测。以已有权重微调时修改 `--weights`；这会开始新的训练过程，不恢复旧优化器状态。

## 测试与限制

`tests/` 覆盖 README 中的功能、桌面样例和证据核验，保留为离线回归测试；`evidence/` 保存 README 引用的结果。一次性本地验证脚本和视频制作中间文件不属于发布源码。

```powershell
# 代码回归测试
.\.venv\Scripts\python.exe -m tools.test

# README 相邻气泡扫描微基准：仅需 Python，无需模型或 CUDA
py -3.12 -B tools/benchmark_fragments.py

# 核对 README 数字与保存的证据
py -3.12 -B evidence/verify_claims.py

# 准备好模型缓存后测量真实推理耗时
.\.venv\Scripts\python.exe tools/benchmark.py ".\input\page.jpg" --rounds 3 --translator vntl --offline
```

扫描微基准分别检查 20、200、2,000 个框，预热后计时 5 次，输出中位数，并验证配对结果和原始字段不变。数据构造与校验不计入耗时；实测速度取决于机器，不要求与 README 的历史记录完全相同。
