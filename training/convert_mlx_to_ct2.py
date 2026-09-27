"""Exportă modelul adaptat medical (antrenat în MLX) pentru serverul de referință: HuggingFace -> CTranslate2.

Pornim de la modelul HuggingFace original FraPiz și înlocuim toți tensorii cu cei din modelul MLX antrenat
(decoder, straturile de encoder deblocate, eventual media WiSE-FT). Rezultatul rulează cu faster-whisper (CUDA/CPU).

    python -m training.convert_mlx_to_ct2 models/mlx-whisper-turbo-moldovan-med \\
        --hf models/hf-whisper-turbo-moldovan --out models/ct2-whisper-turbo-moldovan-med
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.convert_hf_to_mlx import map_name  # noqa: E402


def main() -> None:
    import mlx.core as mx
    from safetensors.numpy import save_file

    ap = argparse.ArgumentParser()
    ap.add_argument("mlx_model")
    ap.add_argument("--hf", default="models/hf-whisper-turbo-moldovan")
    ap.add_argument("--out", default="models/ct2-whisper-turbo-moldovan-med")
    ap.add_argument("--quantization", default="float16")
    a = ap.parse_args()
    src, hf_dir, out = Path(a.mlx_model), Path(a.hf), Path(a.out)

    trained = mx.load(str(src / "weights.safetensors"))
    hf_tmp = out.with_name(out.name + "-hf")
    if hf_tmp.exists():
        shutil.rmtree(hf_tmp)
    shutil.copytree(hf_dir, hf_tmp)
    replaced = 0
    for f in sorted(hf_tmp.glob("*.safetensors")):
        w = {k: np.array(v.astype(mx.float32)) for k, v in mx.load(str(f)).items()}
        for k in w:
            n = map_name(k)
            if n and n in trained and n != "encoder._positional_embedding":
                v = np.array(trained[n].astype(mx.float32))
                if n.endswith("conv1.weight") or n.endswith("conv2.weight"):
                    v = np.transpose(v, (0, 2, 1))  # MLX (out, k, in) -> PyTorch (out, in, k)
                if v.shape != w[k].shape:
                    raise SystemExit(f"Formă diferită pentru {k}: {v.shape} vs {w[k].shape}")
                w[k] = v
                replaced += 1
        save_file({k: v.astype(np.float16) for k, v in w.items()}, str(f), metadata={"format": "pt"})
    print(f"{replaced} tensori înlocuiți -> {hf_tmp}", file=sys.stderr)

    if out.exists():
        shutil.rmtree(out)
    subprocess.run([str(Path(sys.executable).parent / "ct2-transformers-converter"), "--model", str(hf_tmp),
                    "--output_dir", str(out), "--quantization", a.quantization,
                    "--copy_files", "tokenizer.json", "preprocessor_config.json"], check=True)
    shutil.rmtree(hf_tmp)
    (out / "SOURCE.json").write_text(json.dumps({"converted_from": str(src), "base_hf": str(hf_dir),
                                                "decoder_tensors_replaced": replaced}))
    print(f"-> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
