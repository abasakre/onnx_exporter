#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
# flake8: noqa

"""
Note: Code in this file is modified from
https://github.com/TadaoYamaoka/whisper/blob/main/to_onnx.py

Thanks to https://github.com/TadaoYamaoka
for making the onnx export script public.

Note that we have removed the 30 seconds constraint from whisper. You can
use any T <= 30.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import onnx
import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn

import whisper
from whisper.model import (
    AudioEncoder,
    MultiHeadAttention,
    ResidualAttentionBlock,
    TextDecoder,
)


def get_args():
    parser = argparse.ArgumentParser(description="Export Whisper ASR models to ONNX and INT8 for Kurdish & offline sherpa-onnx.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["tiny", "base", "small", "all"],
        help="Model size to export (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./exported_asr_models",
        help="Target output directory (default: ./exported_asr_models)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization",
    )
    return parser.parse_args()


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    if "large" in filename or "turbo" in filename:
        external_filename = filename.split(".onnx")[0]
        onnx.save(
            model,
            filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_filename + ".weights",
        )
    else:
        onnx.save(model, filename)


def modified_audio_encoder_forward(self: AudioEncoder, x: torch.Tensor):
    """
    x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        the mel spectrogram of the audio
    """
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    if False:
        # This branch contains the original code
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
    else:
        # This branch contains the actual changes
        assert (
            x.shape[2] == self.positional_embedding.shape[1]
        ), f"incorrect audio shape: {x.shape}, {self.positional_embedding.shape}"
        assert (
            x.shape[1] == self.positional_embedding.shape[0]
        ), f"incorrect audio shape: {x.shape}, {self.positional_embedding.shape}"
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

    for block in self.blocks:
        x = block(x)

    x = self.ln_post(x)
    return x


AudioEncoder.forward = modified_audio_encoder_forward


class AudioEncoderTensorCache(nn.Module):
    def __init__(self, inAudioEncoder: AudioEncoder, inTextDecoder: TextDecoder):
        super().__init__()
        self.audioEncoder = inAudioEncoder
        self.textDecoder = inTextDecoder

    def forward(self, x: Tensor):
        audio_features = self.audioEncoder(x)

        n_layer_cross_k_list = []
        n_layer_cross_v_list = []
        for block in self.textDecoder.blocks:
            n_layer_cross_k_list.append(block.cross_attn.key(audio_features))
            n_layer_cross_v_list.append(block.cross_attn.value(audio_features))

        return torch.stack(n_layer_cross_k_list), torch.stack(n_layer_cross_v_list)


class MultiHeadAttentionCross(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.multiHeadAttention.query(x)
        wv, qk = self.multiHeadAttention.qkv_attention(q, k, v, mask)
        return self.multiHeadAttention.out(wv)


class MultiHeadAttentionSelf(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,  # (b, n_ctx      , n_state)
        k_cache: Tensor,  # (b, n_ctx_cache, n_state)
        v_cache: Tensor,  # (b, n_ctx_cache, n_state)
        mask: Tensor,
    ):
        q = self.multiHeadAttention.query(x)  # (b, n_ctx, n_state)
        k = self.multiHeadAttention.key(x)  # (b, n_ctx, n_state)
        v = self.multiHeadAttention.value(x)  # (b, n_ctx, n_state)

        k_cache[:, -k.shape[1] :, :] = k  # (b, n_ctx_cache + n_ctx, n_state)
        v_cache[:, -v.shape[1] :, :] = v  # (b, n_ctx_cache + n_ctx, n_state)

        wv, qk = self.multiHeadAttention.qkv_attention(q, k_cache, v_cache, mask)
        return self.multiHeadAttention.out(wv), k_cache, v_cache


class ResidualAttentionBlockTensorCache(nn.Module):
    def __init__(self, inResidualAttentionBlock: ResidualAttentionBlock):
        super().__init__()
        self.originalBlock = inResidualAttentionBlock
        self.attn = MultiHeadAttentionSelf(inResidualAttentionBlock.attn)
        self.cross_attn = (
            MultiHeadAttentionCross(inResidualAttentionBlock.cross_attn)
            if inResidualAttentionBlock.cross_attn
            else None
        )

    def forward(
        self,
        x: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        mask: Tensor,
    ):
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.attn(
            self.originalBlock.attn_ln(x), self_k_cache, self_v_cache, mask=mask
        )
        x = x + self_attn_x

        if self.cross_attn:
            x = x + self.cross_attn(
                self.originalBlock.cross_attn_ln(x), cross_k, cross_v
            )

        x = x + self.originalBlock.mlp(self.originalBlock.mlp_ln(x))
        return x, self_k_cache_updated, self_v_cache_updated


class TextDecoderTensorCache(nn.Module):
    def __init__(self, inTextDecoder: TextDecoder, in_n_ctx: int):
        super().__init__()
        self.textDecoder = inTextDecoder
        self.n_ctx = in_n_ctx

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlockTensorCache(orginal_block))

    def forward(
        self,
        tokens: Tensor,
        n_layer_self_k_cache: Tensor,
        n_layer_self_v_cache: Tensor,
        n_layer_cross_k: Tensor,
        n_layer_cross_v: Tensor,
        offset: Tensor,
    ):
        x = (
            self.textDecoder.token_embedding(tokens)
            + self.textDecoder.positional_embedding[
                offset[0] : offset[0] + tokens.shape[-1]
            ]
        )
        x = x.to(n_layer_cross_k[0].dtype)

        i = 0
        for block in self.blocks:
            self_k_cache = n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :]
            self_v_cache = n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :]
            x, self_k_cache, self_v_cache = block(
                x,
                self_k_cache=self_k_cache,
                self_v_cache=self_v_cache,
                cross_k=n_layer_cross_k[i],
                cross_v=n_layer_cross_v[i],
                mask=self.textDecoder.mask,
            )
            n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_k_cache
            n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_v_cache
            i += 1

        x = self.textDecoder.ln(x)

        if False:
            # x.shape (1, 3, 384)
            # weight.shape (51684, 384)

            logits = (
                x
                @ torch.transpose(
                    self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1
                )
            ).float()
        else:
            logits = (
                torch.matmul(
                    self.textDecoder.token_embedding.weight.to(x.dtype),
                    x.permute(0, 2, 1),
                )
                .permute(0, 2, 1)
                .float()
            )

        return logits, n_layer_self_k_cache, n_layer_self_v_cache


# ref: https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-pt-to-ggml.py#L232
def convert_tokens(name, model):
    whisper_dir = Path(whisper.__file__).parent
    multilingual = model.is_multilingual
    tokenizer = (
        whisper_dir
        / "assets"
        / (multilingual and "multilingual.tiktoken" or "gpt2.tiktoken")
    )
    if not tokenizer.is_file():
        raise ValueError(f"Cannot find {tokenizer}")

    #  import base64

    tokens_path = os.path.join(output_dir, "tokens.txt")
    with open(tokenizer, "r") as f:
        contents = f.read()
        tokens = {
            token: int(rank)
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }

    with open(tokens_path, "w", encoding="utf-8") as f:
        for t, i in tokens.items():
            f.write(f"{t} {i}\n")
    print(f"  ✓ Saved tokens file: {tokens_path} ({len(tokens)} tokens)")


def load_model(name: str):
    """Load a Whisper model by name."""
    return whisper.load_model(name)


@torch.no_grad()
def export_whisper_model(name: str, base_output_dir: str, quantize: bool = True):
    target_dir = os.path.join(base_output_dir, f"whisper-{name}")
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"🚀 Exporting Whisper ASR ({name.upper()}) to ONNX")
    print(f"=======================================================")

    opset_version = 17

    print("1. Loading Whisper PyTorch model...")
    model = load_model(name)

    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"   Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")

    # Write tokens.txt
    print("2. Writing tokens mapping...")
    convert_tokens(output_dir=target_dir, model=model)

    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )

    model.eval()
    audio = torch.rand(16000 * 2)
    audio = whisper.pad_or_trim(audio)

    n_mels = 128 if name in ("distil-large-v3", "distil-large-v3.5", "large", "large-v3", "turbo") else 80

    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(model.device).unsqueeze(0)
    batch_size = 1

    # 3. Export Encoder
    print("3. Tracing and exporting Encoder...")
    encoder = AudioEncoderTensorCache(model.encoder, model.decoder)
    encoder_filename = os.path.join(target_dir, "encoder.onnx")

    torch.onnx.export(
        encoder,
        mel,
        encoder_filename,
        opset_version=opset_version,
        input_names=["mel"],
        output_names=["n_layer_cross_k", "n_layer_cross_v"],
        dynamic_axes={
            "mel": {0: "n_audio", 2: "T"},
            "n_layer_cross_k": {1: "n_audio", 2: "T"},
            "n_layer_cross_v": {1: "n_audio", 2: "T"},
        },
    )

    encoder_meta_data = {
        "model_type": f"whisper-{name}",
        "version": "1",
        "maintainer": "k2-fsa",
        "n_mels": model.dims.n_mels,
        "n_audio_ctx": model.dims.n_audio_ctx,
        "n_audio_state": model.dims.n_audio_state,
        "n_audio_head": model.dims.n_audio_head,
        "n_audio_layer": model.dims.n_audio_layer,
        "n_vocab": model.dims.n_vocab,
        "n_text_ctx": model.dims.n_text_ctx,
        "n_text_state": model.dims.n_text_state,
        "n_text_head": model.dims.n_text_head,
        "n_text_layer": model.dims.n_text_layer,
        "sot_sequence": ",".join(list(map(str, tokenizer.sot_sequence))),
        "all_language_tokens": ",".join(list(map(str, tokenizer.all_language_tokens))),
        "all_language_codes": ",".join(tokenizer.all_language_codes),
        "sot": tokenizer.sot,
        "sot_index": tokenizer.sot_sequence.index(tokenizer.sot),
        "eot": tokenizer.eot,
        "blank_id": tokenizer.encode(" ")[0],
        "is_multilingual": int(model.is_multilingual),
        "no_speech": tokenizer.no_speech,
        "non_speech_tokens": ",".join(list(map(str, tokenizer.non_speech_tokens))),
        "transcribe": tokenizer.transcribe,
        "translate": tokenizer.translate,
        "sot_prev": tokenizer.sot_prev,
        "sot_lm": tokenizer.sot_lm,
        "no_timestamps": tokenizer.no_timestamps,
    }
    add_meta_data(filename=encoder_filename, meta_data=encoder_meta_data)
    enc_size_mb = os.path.getsize(encoder_filename) / (1024 * 1024)
    print(f"  ✓ Exported encoder.onnx: {enc_size_mb:.2f} MB")

    # 4. Export Decoder
    print("4. Tracing and exporting Decoder...")
    n_layer_cross_k, n_layer_cross_v = encoder(mel)
    n_audio = mel.shape[0]
    tokens = torch.tensor([[tokenizer.sot, tokenizer.sot, tokenizer.sot]] * n_audio).to(mel.device)
    decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)
    n_layer_self_k_cache = torch.zeros(
        (len(model.decoder.blocks), n_audio, model.dims.n_text_ctx, model.dims.n_text_state),
        device=mel.device,
    )
    n_layer_self_v_cache = torch.zeros(
        (len(model.decoder.blocks), n_audio, model.dims.n_text_ctx, model.dims.n_text_state),
        device=mel.device,
    )
    offset = torch.zeros(1, dtype=torch.int64).to(mel.device)

    decoder_filename = os.path.join(target_dir, "decoder.onnx")
    torch.onnx.export(
        decoder,
        (tokens, n_layer_self_k_cache, n_layer_self_v_cache, n_layer_cross_k, n_layer_cross_v, offset),
        decoder_filename,
        opset_version=opset_version,
        input_names=[
            "tokens",
            "in_n_layer_self_k_cache",
            "in_n_layer_self_v_cache",
            "n_layer_cross_k",
            "n_layer_cross_v",
            "offset",
        ],
        output_names=["logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache"],
        dynamic_axes={
            "tokens": {0: "n_audio", 1: "n_tokens"},
            "in_n_layer_self_k_cache": {1: "n_audio"},
            "in_n_layer_self_v_cache": {1: "n_audio"},
            "n_layer_cross_k": {1: "n_audio", 2: "T"},
            "n_layer_cross_v": {1: "n_audio", 2: "T"},
        },
    )
    dec_size_mb = os.path.getsize(decoder_filename) / (1024 * 1024)
    print(f"  ✓ Exported decoder.onnx: {dec_size_mb:.2f} MB")

    # 5. INT8 Quantization
    if quantize:
        print("5. Generating INT8 Quantized models...")
        enc_int8 = os.path.join(target_dir, "encoder.int8.onnx")
        quantize_dynamic(
            model_input=encoder_filename,
            model_output=enc_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
        enc_int8_mb = os.path.getsize(enc_int8) / (1024 * 1024)

        dec_int8 = os.path.join(target_dir, "decoder.int8.onnx")
        quantize_dynamic(
            model_input=decoder_filename,
            model_output=dec_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
        dec_int8_mb = os.path.getsize(dec_int8) / (1024 * 1024)

        total_int8_mb = enc_int8_mb + dec_int8_mb
        print(f"  ✓ Created encoder.int8.onnx: {enc_int8_mb:.2f} MB")
        print(f"  ✓ Created decoder.int8.onnx: {dec_int8_mb:.2f} MB")
        print(f"  📦 Total INT8 model package size: {total_int8_mb:.2f} MB")

    # Save model_config.json
    config_data = {
        "model_type": "whisper",
        "model_size": name,
        "sampling_rate": 16000,
        "multilingual": True,
        "language_code": "ku",
        "kurdish_supported": True,
        "files": {
            "encoder": "encoder.int8.onnx" if quantize else "encoder.onnx",
            "decoder": "decoder.int8.onnx" if quantize else "decoder.onnx",
            "tokens": "tokens.txt",
        },
    }
    import json
    with open(os.path.join(target_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"✨ Whisper {name} exported successfully to: {target_dir}")


def main():
    args = get_args()
    models = ["tiny", "base"] if args.model == "all" else [args.model]

    for m in models:
        export_whisper_model(
            name=m,
            base_output_dir=args.output_dir,
            quantize=not args.no_quantize,
        )

    print("\n🎉 All requested Whisper ASR models exported successfully!")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from whisper.model import disable_sdpa

    with disable_sdpa():
        main()
