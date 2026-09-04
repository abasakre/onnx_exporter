#!/usr/bin/env python3
"""
Kurdish TTS (VITS / MMS) to ONNX Converter and Quantizer.

Converts Hugging Face models:
  - Sorani: akam-ot/ckb-tts
  - Badini: akam-ot/mms-tts-kmr-arabic-finetune-base
into optimized ONNX and INT8 quantized models for offline mobile/desktop deployment.
"""

import argparse
import json
import os
import sys
import shutil
import numpy as np

from huggingface_hub import hf_hub_download
import torch
from transformers import AutoTokenizer, VitsModel
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import soundfile as sf

MODEL_REGISTRY = {
    "sorani": {
        "repo_id": "akam-ot/ckb-tts",
        "name": "ckb-tts",
        "sample_text": "سڵاو چۆنی، هیوادارم باش بیت",
        "type": "vits",
    },
    "badini": {
        "repo_id": "akam-ot/mms-tts-kmr-arabic-finetune-base",
        "name": "kmr-tts",
        "sample_text": "سلاڤ، ئەز نها ب کوردی دئاخڤم",
        "type": "vits",
    },
    "kurmanji": {
        "repo_id": "rojanu/piper-ku-berfin-renas-medium",
        "name": "kurmanji-piper",
        "sample_text": "Slav, ez niha bi kurdî diaxivim",
        "type": "piper",
        "onnx_filename": "ku-berfin_renas-medium.onnx",
        "json_filename": "ku-berfin_renas-medium.onnx.json",
    },
}


def normalize_kurdish_text(text: str) -> str:
    """Kurdish normalization matching the Hugging Face space."""
    text = text.replace("\u06a9", "\u0643")  # Kurdish ک -> Arabic ك
    for zw in ["\u200c", "\u200d", "\u200b"]:
        text = text.replace(zw, "")
    return text.strip()


class VitsInferenceWrapper(torch.nn.Module):
    """Wraps VitsModel to output only the synthesized audio waveform."""

    def __init__(self, model: VitsModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.waveform


def export_tokens_file(tokenizer, output_path: str):
    """Exports tokens.txt for sherpa-onnx and onnx runtimes."""
    vocab = tokenizer.get_vocab()
    # Sort by token ID
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])

    with open(output_path, "w", encoding="utf-8") as f:
        for token, idx in sorted_tokens:
            f.write(f"{token} {idx}\n")
    print(f"  ✓ Saved tokens file: {output_path} ({len(sorted_tokens)} tokens)")


def export_dialect(dialect_key: str, base_output_dir: str, quantize: bool = True, verify: bool = True):
    info = MODEL_REGISTRY[dialect_key]
    repo_id = info["repo_id"]
    model_type = info.get("type", "vits")
    target_dir = os.path.join(base_output_dir, dialect_key)
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"🚀 Exporting {dialect_key.upper()} ({repo_id})")
    print(f"=======================================================")

    onnx_path = os.path.join(target_dir, "model.onnx")

    if model_type == "piper":
        print(f"1. Downloading Piper ONNX model & config from Hugging Face ({repo_id})...")
        downloaded_onnx = hf_hub_download(repo_id=repo_id, filename=info["onnx_filename"])
        downloaded_json = hf_hub_download(repo_id=repo_id, filename=info["json_filename"])

        with open(downloaded_json, "r", encoding="utf-8") as f:
            piper_cfg = json.load(f)

        sample_rate = piper_cfg.get("audio", {}).get("sample_rate", 22050)
        speakers = piper_cfg.get("speaker_id_map", {"Berfin": 0, "Renas": 1})
        print(f"   Sampling Rate: {sample_rate} Hz")
        print(f"   Available Speakers: {speakers}")

        shutil.copyfile(downloaded_onnx, onnx_path)
        shutil.copyfile(downloaded_json, os.path.join(target_dir, "model.onnx.json"))

        # Generate tokens.txt from phoneme_id_map
        tokens_file = os.path.join(target_dir, "tokens.txt")
        phoneme_id_map = piper_cfg.get("phoneme_id_map", {})
        sorted_tokens = []
        for phoneme, ids in phoneme_id_map.items():
            for i in ids:
                sorted_tokens.append((phoneme, i))
        sorted_tokens.sort(key=lambda x: x[1])

        with open(tokens_file, "w", encoding="utf-8") as f:
            for token, idx in sorted_tokens:
                f.write(f"{token} {idx}\n")
        print(f"  ✓ Saved tokens file: {tokens_file} ({len(sorted_tokens)} tokens)")

        # Save metadata config
        meta_config = {
            "model_id": repo_id,
            "dialect": dialect_key,
            "sampling_rate": sample_rate,
            "model_type": "piper",
            "speakers": speakers,
            "phoneme_type": piper_cfg.get("phoneme_type", "espeak"),
            "espeak_voice": piper_cfg.get("espeak", {}).get("voice", "ku"),
        }
        with open(os.path.join(target_dir, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(meta_config, f, indent=2, ensure_ascii=False)

        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ✓ Model ONNX ready: {onnx_size_mb:.2f} MB")

    else:
        # Standard Hugging Face VitsModel export
        print("1. Downloading model and tokenizer from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = VitsModel.from_pretrained(repo_id).eval()
        wrapper = VitsInferenceWrapper(model).eval()

        sample_rate = model.config.sampling_rate
        print(f"   Sampling Rate: {sample_rate} Hz")

        # Save tokens.txt and config.json
        tokens_file = os.path.join(target_dir, "tokens.txt")
        export_tokens_file(tokenizer, tokens_file)

        meta_config = {
            "model_id": repo_id,
            "dialect": dialect_key,
            "sampling_rate": sample_rate,
            "model_type": "vits",
        }
        with open(os.path.join(target_dir, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(meta_config, f, indent=2, ensure_ascii=False)

        # Prepare dummy input for tracing
        sample_text = normalize_kurdish_text(info["sample_text"])
        inputs = tokenizer(sample_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        print(f"2. Tracing and exporting to ONNX format: {onnx_path} ...")

        with torch.no_grad():
            export_kwargs = {
                "export_params": True,
                "opset_version": 16,
                "do_constant_folding": True,
                "input_names": ["input_ids", "attention_mask"],
                "output_names": ["waveform"],
                "dynamic_axes": {
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "waveform": {0: "batch_size", 2: "audio_samples"},
                },
            }
            # In PyTorch 2.5+, Dynamo exporter is used by default which fails on VITS ops (aten._is_all_true).
            # We explicitly enforce the stable TorchScript exporter (dynamo=False).
            import inspect
            if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                export_kwargs["dynamo"] = False

            torch.onnx.export(
                wrapper,
                (input_ids, attention_mask),
                onnx_path,
                **export_kwargs,
            )

        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ✓ Exported model.onnx: {onnx_size_mb:.2f} MB")

    # Check and simplify model
    print("3. Validating ONNX model integrity...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX checker passed successfully.")

    # Quantize to INT8
    int8_path = os.path.join(target_dir, "model.int8.onnx")
    if quantize:
        print(f"4. Performing dynamic INT8 quantization -> {int8_path} ...")
        quantize_dynamic(
            model_input=onnx_path,
            model_output=int8_path,
            weight_type=QuantType.QUInt8,
        )
        int8_size_mb = os.path.getsize(int8_path) / (1024 * 1024)
        savings = ((onnx_size_mb - int8_size_mb) / onnx_size_mb) * 100
        print(f"  ✓ Created quantized model: {int8_size_mb:.2f} MB ({savings:.1f}% reduction)")

    # Verification with ONNX Runtime
    if verify:
        test_model_path = int8_path if quantize else onnx_path
        print(f"5. Verifying inference with ONNX Runtime ({os.path.basename(test_model_path)})...")
        session = ort.InferenceSession(test_model_path, providers=["CPUExecutionProvider"])

        if model_type == "piper":
            # Test inference with sample phoneme IDs (Slav)
            phonemes = np.array([[1, 31, 24, 14, 34, 2]], dtype=np.int64)
            lengths = np.array([phonemes.shape[1]], dtype=np.int64)
            scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
            sid = np.array([0], dtype=np.int64)
            ort_inputs = {
                "input": phonemes,
                "input_lengths": lengths,
                "scales": scales,
                "sid": sid,
            }
            ort_outputs = session.run(None, ort_inputs)
            waveform = np.squeeze(ort_outputs[0])
        else:
            test_text = normalize_kurdish_text(info["sample_text"])
            test_inputs = tokenizer(test_text, return_tensors="np")

            ort_inputs = {
                "input_ids": test_inputs["input_ids"].astype(np.int64),
                "attention_mask": test_inputs.get("attention_mask", np.ones_like(test_inputs["input_ids"])).astype(np.int64),
            }

            ort_outputs = session.run(None, ort_inputs)
            waveform = np.squeeze(ort_outputs[0])

        # Normalize waveform
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak * 0.9

        test_wav = os.path.join(target_dir, "sample_test.wav")
        sf.write(test_wav, waveform, sample_rate)
        print(f"  ✓ Verification passed! Synthesized test audio: {test_wav}")

    print(f"✨ Successfully exported {dialect_key} to {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export Kurdish TTS models to ONNX and INT8.")
    parser.add_argument(
        "--dialect",
        choices=["sorani", "badini", "kurmanji", "all"],
        default="all",
        help="Which dialect to export (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="./exported_models",
        help="Target output directory (default: ./exported_models)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX Runtime verification test",
    )

    args = parser.parse_args()

    dialects = ["sorani", "badini", "kurmanji"] if args.dialect == "all" else [args.dialect]

    for d in dialects:
        export_dialect(
            dialect_key=d,
            base_output_dir=args.output_dir,
            quantize=not args.no_quantize,
            verify=not args.no_verify,
        )

    print("\n🎉 All requested models exported successfully!")
    print(f"Check your output directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
