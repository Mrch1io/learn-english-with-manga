# Usage Guide

**Language / 语言**: English | [简体中文](USAGE.md)

[Back to the project home page](README.md)

Supports Japanese-to-English manga translation, manual proofreading and Japanese click-to-listen. The steps below apply to Windows + NVIDIA CUDA, and all commands are run from the project root; replace the example input paths with your own files.

## Installation

Create an environment with Python 3.12:

```powershell
py -3.12 -m venv .venv
```

Install the project dependencies directly from [requirements.txt](requirements.txt), which pins the CUDA 12.6 builds of PyTorch (`2.10.0+cu126`) and torchvision (`0.25.0+cu126`). No separate PyTorch installation is needed.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'"
```

Bubble detection uses `best.pt` in the project root. Prepare the other models and runtimes as listed below; download sources, pinned versions and licenses are in the [third-party notices](THIRD_PARTY.md).

| Resource | Location and purpose |
|---|---|
| VNTL GGUF | `runtime/vntl/vntl-llama3-8b-v2-hf.Q4_K_M.gguf`, the default translation model |
| CUDA llama.cpp b11706 | `runtime/llama.cpp/b11706/llama-server.exe`; the program manages the translation server |
| Manga-OCR, LaMa | Downloaded and cached the first time they're needed; used for Japanese recognition and for text removal on complex backgrounds, respectively |
| Magi v2 | `runtime/magiv2`, used by default to detect text outside bubbles; turn it off with `--bubbles-only` |
| VOICEVOX | Optional, needed only for Japanese click-to-listen; see [voice setup](#japanese-click-to-listen-and-voices) |

The default detection pipeline also needs the Magi dependencies and weights:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-magi.txt
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('ragavsachdeva/magiv2', revision='fbc890fec52977142e8ee00bfe26e9458b65517c', local_dir='runtime/magiv2')"
```

Magi is for academic research use only. The optional Qwen backend also needs the `Qwen/Qwen2.5-3B-Instruct` base model and `manga-lora-adapter/` in the project root; the default VNTL doesn't need them.

## Launching and Translating

```powershell
.\launch.bat
```

After launch, the reader opens with the three-page sample included in the repository. The sample's English and characters were curated by hand to demonstrate the workflow; viewing the sample loads no inference models, and playback requires VOICEVOX.

- **Page-by-page translation**: Import through **Import → Images / JSON... / Folder...**, set the work name and output directory in **Tools → Processing**, and click **Tools → Translate Current Page** to generate the translated image.
- **Batch translation**: Go to **Tools → Tasks**, choose images or a folder, the output directory and the work name, and click **Start Translation**. Check **Text Only** to save the recognized and translated text first.
- **Reading results**: Switch images with **Show Original / Show Translation**. Existing text-only results need **Tools → Export Saved Translation** to generate translated images.

The launcher prefers `.venv\Scripts\python.exe` and otherwise reads the interpreter path from `.gpu-python`; both launch methods require CUDA.

## Command Line

```powershell
# Process a folder recursively, with pages in natural order
.\run.bat ".\input" -o ".\translated_output" --work "Book1"

# Save the text first, then proofread or export it in the reader later
.\run.bat ".\input" -o ".\text_output" --work "Book1" --text-only
.\run.bat --read ".\text_output"

# Skip Magi and process only the text inside bubbles
.\run.bat ".\input\page.jpg" -o ".\translated_output" --work "Book1" --bubbles-only

# Show all options
.\.venv\Scripts\python.exe main.py --help
```

Translation requires an input path. The same command can resume a run; manual revisions are kept by default, and pages are marked for review when the configuration changes. `--force` allows regenerating and overwriting manual revisions. Keep the original images, as well as the matching JSON, PNG and `.clean.png` in the results; terms and character information are stored separately in `data/memory.sqlite3`.

Common options:

| Option | Purpose |
|---|---|
| `--offline` | Use only existing weights and caches; download them first |
| `--device cuda:1` | Select the GPU, default `cuda:0`; CPU inference is not supported |
| `--memory-policy staged` | Default staged VRAM management; `resident` is for setups with ample VRAM |
| `--translator vntl` | Default backend; `qwen` uses the separately prepared base model and LoRA |
| `--gguf` / `--llama-server` | Specify the translation weights or the llama.cpp executable |
| `--mode individual` | Translate box by box; by default, VNTL groups boxes automatically and Qwen goes box by box |
| `--no-speaker-context` | Turn off confirmed character information in the translation context |
| `--font` | Specify a TTF, OTF or TTC font that supports the required characters |

## Proofreading and Terminology

**Workbench** at the top of the reader opens proofreading. **Save Translation & Redraw** edits the English; **Save Source & Retranslate** edits the Japanese and retranslates the current box. After editing a text-only result, you still need **Export Image** to export the translated image.

For local OCR, use **Enable Local OCR → Preview OCR Candidate → Apply Candidate & Retranslate**; previewing does not change the source text. To clean up leftover text, use **Tools → Image Cleanup → Clean Selection & Export**. **Tools → Revision Manager** shows the kept revisions and jumps to proofreading.

Terms are isolated per work; locked names keep naming consistent:

```powershell
.\run.bat --work "Book1" --set-term "高梨美咲" "Misaki Takanashi" --lock
.\run.bat --work "Book1" --list-terms
.\run.bat --work "Book1" --export-terms ".\terms.json"
.\run.bat --work "Book1" --import-terms ".\terms.json"
```

## Japanese Click-to-Listen and Voices

Download the Windows DirectML build from the [VOICEVOX Engine 0.25.2 release page](https://github.com/VOICEVOX/voicevox_engine/releases/tag/0.25.2), extract it to `runtime/voicevox/windows-directml/`, and make sure it contains `run.exe`.

In the reader, **Settings → Start / Connect** connects to or starts the engine, **Refresh Voices** updates the voice list, and **Preview Voice** plays a sample. Clicking a bubble reads its saved Japanese aloud; this can be turned off with **Play on bubble click** in the settings. In the side panel, **Bind Selected Bubble** confirms the character, and **Settings → Manage Character → Save Name & Default Voice** saves the name and default voice. Magi's character suggestions need manual confirmation.

The engine listens on `127.0.0.1:50021` by default and keeps running independently after the reader is closed, with logs in `data/voicevox/`. For the CLI, start the engine first, then list voices with `--list-voices`; `--speak-page` only generates WAV files. Playback currently supports Windows; for voice usage and attribution requirements, see the [third-party notices](THIRD_PARTY.md).

## Training the Bubble Detector

Use [tools/train.py](tools/train.py) after preparing your own dataset in Ultralytics format along with `data.yaml`:

```powershell
.\.venv\Scripts\python.exe tools/train.py --data ".\dataset\data.yaml" --weights yolov8m.pt --epochs 30
```

The first training run saves to `output/training/bubble-detector/` by default, and the directory name increments automatically on repeated runs. Put `weights/best.pt` from the actual output into the project root to use it for detection. To fine-tune from existing weights, change `--weights`; this starts a new training run and does not restore the old optimizer state.

## Tests and Limitations

`tests/` covers the features in the README, the desktop sample and evidence verification, and is kept as offline regression tests; `evidence/` stores the results cited in the README. One-off local validation scripts and intermediate video-production files are not part of the released source.

```powershell
# Code regression tests
.\.venv\Scripts\python.exe -m tools.test

# README adjacent-bubble scan microbenchmark: needs only Python, no models or CUDA
py -3.12 -B tools/benchmark_fragments.py

# Check README numbers against the saved evidence
py -3.12 -B evidence/verify_claims.py

# Measure real inference time once the model caches are ready
.\.venv\Scripts\python.exe tools/benchmark.py ".\input\page.jpg" --rounds 3 --translator vntl --offline
```

The scan microbenchmark checks 20, 200 and 2,000 boxes, times 5 runs after warm-up, outputs the medians, and verifies that the pairing results and original fields are unchanged. Data construction and validation are not included in the timing; measured speed depends on the machine and is not expected to match the historical record in the README exactly.
