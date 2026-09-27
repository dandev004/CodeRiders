"""Conversie Whisper din formatul HuggingFace (transformers) în formatul MLX (Apple Silicon) și CTranslate2.

Folosit pentru modelul FraPiz/whisper-large-v3-turbo-moldovan-romanian (Apache 2.0), antrenat pe cele 70 de ore
ale corpusului de vorbire moldovenească. Verifică fiecare tensor față de formele modelului de bază MLX.

    python -m training.convert_hf_to_mlx models/hf-whisper-turbo-moldovan models/mlx-whisper-turbo-moldovan \
        --reference models/mlx-whisper-large-v3-turbo
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

ATTN = {"q_proj": "query", "k_proj": "key", "v_proj": "value", "out_proj": "out"}


def map_name(k: str) -> str | None:
    k = k.removeprefix("model.")
    if k.startswith("proj_out"):
        return None  # legat de token_embedding
    k = k.replace("encoder.embed_positions.weight", "encoder._positional_embedding")
    k = k.replace("decoder.embed_positions.weight", "decoder.positional_embedding")
    k = k.replace("decoder.embed_tokens.weight", "decoder.token_embedding.weight")
    k = k.replace("encoder.layer_norm.", "encoder.ln_post.").replace("decoder.layer_norm.", "decoder.ln.")
    k = re.sub(r"\.layers\.(\d+)\.", r".blocks.\1.", k)
    k = k.replace(".self_attn_layer_norm.", ".attn_ln.").replace(".encoder_attn_layer_norm.", ".cross_attn_ln.")
    k = k.replace(".final_layer_norm.", ".mlp_ln.").replace(".fc1.", ".mlp1.").replace(".fc2.", ".mlp2.")
    k = k.replace(".self_attn.", ".attn.").replace(".encoder_attn.", ".cross_attn.")
    for a, b in ATTN.items():
        k = k.replace(f".{a}.", f".{b}.")
    return k


def main() -> None:
    import mlx.core as mx

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--reference", required=True, help="model MLX de bază (aceeași arhitectură)")
    a = ap.parse_args()
    src, dst, ref = Path(a.src), Path(a.dst), Path(a.reference)

    hf = {}
    for f in sorted(src.glob("*.safetensors")):
        hf.update({k: np.array(v.astype(mx.float32)) for k, v in mx.load(str(f)).items()})
    base = mx.load(str(ref / "weights.safetensors"))
    out = {}
    for k, v in hf.items():
        n = map_name(k)
        if n is None or n == "encoder._positional_embedding":
            continue  # MLX calculează pozițiile encoder-ului (sinusoide), identice cu cele din HF
        if n.endswith("conv1.weight") or n.endswith("conv2.weight"):
            v = np.transpose(v, (0, 2, 1))  # PyTorch (out, in, k) -> MLX (out, k, in)
        if n not in base:
            raise SystemExit(f"Tensor necunoscut: {k} -> {n}")
        if tuple(base[n].shape) != v.shape:
            raise SystemExit(f"Formă diferită pentru {n}: {v.shape} vs {base[n].shape}")
        out[n] = mx.array(v.astype(np.float16))
    missing = [k for k in base if k not in out]
    for k in missing:
        if k != "alignment_heads":
            raise SystemExit(f"Lipsește tensorul {k}")
        out[k] = base[k]
    changed = np.mean([float(mx.abs(out[k].astype(mx.float32) - base[k].astype(mx.float32)).max()) > 0
                       for k in out if k != "alignment_heads"])
    dst.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst / "weights.safetensors"), out)
    shutil.copy(ref / "config.json", dst / "config.json")
    (dst / "SOURCE.json").write_text(json.dumps({"converted_from": str(src), "license": "apache-2.0",
                                                 "tensors": len(out), "tensors_changed_vs_base": round(changed, 3)}))
    print(f"{len(out)} tensori, {changed:.0%} diferiți față de modelul de bază -> {dst}")


if __name__ == "__main__":
    main()
