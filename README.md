# Kurdish TTS to ONNX Exporter & Quantizer

This sub-folder contains tools to convert the Hugging Face Kurdish Text-to-Speech models:
- **Sorani (سۆرانی)**: [`akam-ot/ckb-tts`](https://huggingface.co/akam-ot/ckb-tts)
- **Badini (بادینی)**: [`akam-ot/mms-tts-kmr-arabic-finetune-base`](https://huggingface.co/akam-ot/mms-tts-kmr-arabic-finetune-base)

into optimized **ONNX** and **INT8 Quantized** models for 100% offline inference in Flutter, Mobile, and Desktop applications.

---

## Output Artifacts

For each dialect, the script generates:

| File | Description | Typical Size |
|---|---|---|
| `model.onnx` | Full FP32 ONNX model with dynamic axes | ~330 MB |
| `model.int8.onnx` | Quantized INT8 model for mobile on-device inference | **~85 MB** |
| `tokens.txt` | Vocabulary token mappings for ONNX runtime / sherpa-onnx | ~2 KB |
| `model_config.json` | Metadata and sampling rate (16000 Hz) | < 1 KB |
| `sample_test.wav` | Audio generated during verification step | ~50 KB |

---

## 1. Running Automatically with GitHub Actions (Recommended)

You don't need a powerful machine with PyTorch installed locally. A GitHub Actions workflow is provided in [`.github/workflows/export_onnx.yml`](../.github/workflows/export_onnx.yml).

### How to trigger in GitHub:
1. Push this repository to GitHub.
2. Go to the **Actions** tab on your GitHub repository.
3. Select **"Export Kurdish TTS to ONNX"** workflow.
4. Click **Run workflow** (choose `all`, `sorani`, or `badini`).
5. Once the run finishes (~3–5 minutes), scroll down to **Artifacts** and download:
   - `kurdish-tts-onnx-sorani.zip`
   - `kurdish-tts-onnx-badini.zip`

---

## 2. Running Locally (If you have Python + PyTorch)

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run export for both dialects
python3 export_onnx.py --dialect all --output-dir ./exported_models

# Or export only Sorani:
python3 export_onnx.py --dialect sorani --output-dir ./exported_models
```

---

## 3. How to Use Exported Models in Flutter

Copy `model.int8.onnx` and `tokens.txt` into your Flutter app's `assets/models/`:

```
tts/
 └── assets/
      └── models/
           ├── sorani/
           │    ├── model.int8.onnx
           │    └── tokens.txt
           └── badini/
                ├── model.int8.onnx
                └── tokens.txt
```

In `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/sorani/model.int8.onnx
    - assets/models/sorani/tokens.txt
```
