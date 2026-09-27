"""Diarizare locală: „cine a vorbit când” + numărul de participanți, determinat automat din date.

Metodă (fără pyannote/token HF, 100% offline):
  1. Amprente vocale ECAPA-TDNN (SpeechBrain, antrenat pe VoxCeleb) pe ferestre scurte (1.5 s, pas 0.75 s)
     în interiorul frazelor detectate de VAD — pe audio-ul curățat de zgomot.
  2. Matrice de afinitate (cosinus) între ferestre, curățată prin p-pruning (păstrăm doar vecinii cei mai
     apropiați ai fiecărei ferestre) și simetrizată.
  3. Numărul de vorbitori = eigengap-ul laplacianului normalizat (NME-SC, Park et al. 2019):
     NU e un parametru — rezultă din structura datelor. Se alege automat și p-ul de pruning.
  4. Clustering spectral cu k-ul estimat, netezire temporală (median filter) a etichetelor ferestrelor.
  5. Fiecare frază primește vorbitorul majoritar al ferestrelor ei; o frază în care vorbitorul se schimbă
     clar este împărțită în bucăți (split_segments), ca replicile rapide să nu fie atribuite greșit.
  6. Vorbitorii cu < 2% din timpul de vorbire (tuse, zgomot, suprapuneri) sunt absorbiți în cel mai apropiat.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .config import get_config, resolve

log = logging.getLogger("secure_mom.diarize")
SR = 16000


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpeakerEncoder:
    def __init__(self):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        path = str(resolve(get_config().diarization.model))
        self.torch = torch
        self.device = _device()
        # pretrained_path suprascris -> SpeechBrain citește ponderile de pe disc, nu de pe HuggingFace
        self.model = EncoderClassifier.from_hparams(source=path, savedir=path, run_opts={"device": self.device},
                                                    overrides={"pretrained_path": path})
        self.model.eval()

    def embed(self, chunks: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        n = max(len(c) for c in chunks)
        wav = np.zeros((len(chunks), n), dtype=np.float32)
        lens = np.zeros(len(chunks), dtype=np.float32)
        for i, c in enumerate(chunks):
            wav[i, :len(c)] = c
            lens[i] = len(c) / n
        with torch.inference_mode():
            e = self.model.encode_batch(torch.from_numpy(wav).to(self.device), torch.from_numpy(lens).to(self.device))
            e = e.squeeze(1).float().cpu().numpy()
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------
# Clustering spectral cu estimarea numărului de vorbitori (NME-SC)
# ------------------------------------------------------------------------------------------------

def _prune(A: np.ndarray, p: int) -> np.ndarray:
    B = A.copy()
    idx = np.argsort(B, axis=1)[:, :-p]  # tot în afară de cei mai apropiați p vecini
    np.put_along_axis(B, idx, 0.0, axis=1)
    return (B + B.T) / 2


def _eig(A: np.ndarray, kmax: int):
    d = A.sum(axis=1)
    d[d == 0] = 1e-9
    Dm = np.diag(1 / np.sqrt(d))
    L = np.eye(len(A)) - Dm @ A @ Dm
    w, v = np.linalg.eigh(L)
    return w[:kmax + 1], v[:, :kmax + 1]


def estimate_and_cluster(E: np.ndarray, kmax: int = 12, k_fixed: int | None = None) -> np.ndarray:
    from sklearn.cluster import KMeans

    n = len(E)
    if n < 4 or k_fixed == 1:
        return np.zeros(n, dtype=int)
    A = np.clip(E @ E.T, 0, 1)
    np.fill_diagonal(A, 0)
    kmax = min(max(kmax, k_fixed or 0), n - 1)
    best = None
    # alegem p-ul (densitatea grafului) care maximizează eigengap-ul normalizat — criteriul NME
    for frac in (0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.25):
        p = max(2, int(frac * n))
        w, _ = _eig(_prune(A, p), kmax)
        gaps = np.diff(w)
        k = int(np.argmax(gaps[:kmax])) + 1
        ratio = (p / n) / (gaps[k - 1] / (w[-1] + 1e-9) + 1e-9)
        if best is None or ratio < best[0]:
            best = (ratio, p, k)
    _, p, k = best
    if k_fixed:
        k = min(k_fixed, n - 1)  # numărul de participanți indicat de utilizator (opțional)
    if k == 1:
        return np.zeros(n, dtype=int)
    _, v = _eig(_prune(A, p), k)
    X = v[:, :k]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)


def _median_smooth(labels: np.ndarray, groups: np.ndarray, w: int = 3) -> np.ndarray:
    """Netezire în interiorul fiecărei fraze: o fereastră izolată diferită de vecinii ei = zgomot."""
    out = labels.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for j, i in enumerate(idx):
            lo, hi = max(0, j - w // 2), min(len(idx), j + w // 2 + 1)
            vals = labels[idx[lo:hi]]
            out[i] = np.bincount(vals).argmax()
    return out


def diarize(audio_clean: np.ndarray, spans: list[tuple[float, float]],
            num_speakers: int | None = None) -> tuple[list[list[tuple[float, float, str]]], dict]:
    """Pentru fiecare frază: lista de sub-intervale (start, end, vorbitor). De regulă un singur element."""
    cfg = get_config().diarization
    t0 = time.time()
    if not spans:
        return [], {"speakers": 0}
    win, step = float(cfg.window_s), float(cfg.step_s)
    W, S = int(win * SR), int(step * SR)

    windows, owner, wtimes = [], [], []
    for i, (s, e) in enumerate(spans):
        a, b = int(s * SR), int(e * SR)
        if b - a < int(0.5 * SR):
            continue  # prea scurt pentru o amprentă sigură — atribuit ulterior
        starts = list(range(a, max(a + 1, b - W + 1), S)) or [a]
        if b - (starts[-1] + W) > S // 2:
            starts.append(max(a, b - W))
        for k in starts:
            windows.append(audio_clean[k:min(b, k + W)])
            owner.append(i)
            wtimes.append((k / SR, min(b, k + W) / SR))
    owner = np.array(owner)
    if len(windows) < 4:
        return [[(s, e, "S1")] for s, e in spans], {"speakers": 1, "seconds": round(time.time() - t0, 1)}

    enc = SpeakerEncoder()
    E = np.concatenate([enc.embed(windows[b:b + 128]) for b in range(0, len(windows), 128)])
    lab = estimate_and_cluster(E, int(cfg.get("max_speakers", 12)), num_speakers)
    lab = _median_smooth(lab, owner)

    # absorbim vorbitorii cu prea puțină vorbire
    min_share = float(cfg.min_speaker_share)
    while True:
        ids = list(np.unique(lab))
        share = {k: (lab == k).mean() for k in ids}
        small = [k for k in ids if share[k] < min_share]
        if not small or len(ids) <= 1 or num_speakers and len(ids) <= num_speakers:
            break
        k = min(small, key=lambda x: share[x])
        C = {j: E[lab == j].mean(axis=0) for j in ids if j != k}
        ck = E[lab == k].mean(axis=0)
        lab[lab == k] = max(C, key=lambda j: float(C[j] @ ck))

    centroids = {k: E[lab == k].mean(axis=0) for k in np.unique(lab)}
    for k in centroids:
        centroids[k] /= np.linalg.norm(centroids[k]) + 1e-9

    # fraze -> sub-intervale pe vorbitor
    min_piece = float(cfg.get("min_split_s", 2.0))
    raw: list[list[tuple[float, float, int]]] = []
    for i, (s, e) in enumerate(spans):
        idx = np.where(owner == i)[0]
        if len(idx) == 0:
            raw.append([(s, e, -1)])
            continue
        pieces: list[list] = []
        for j in idx:
            ws, we = wtimes[j]
            if pieces and pieces[-1][2] == lab[j]:
                pieces[-1][1] = we
            else:
                pieces.append([ws, we, lab[j]])
        # bucățile prea scurte sunt lipite de vecinul lor
        merged: list[list] = []
        for pc in pieces:
            if merged and (pc[1] - pc[0] < min_piece or merged[-1][2] == pc[2]):
                merged[-1][1] = pc[1]
            else:
                merged.append(pc)
        if len(merged) > 1 and merged[0][1] - merged[0][0] < min_piece:
            merged[1][0] = merged[0][0]
            merged.pop(0)
        merged[0][0], merged[-1][1] = s, e
        for a_, b_ in zip(merged, merged[1:]):
            a_[1] = b_[0] = (a_[1] + b_[0]) / 2 if a_[1] > b_[0] else b_[0]
        raw.append([(a_[0], a_[1], int(a_[2])) for a_ in merged])

    # frazele fără ferestre (foarte scurte) — vorbitorul cel mai apropiat acustic
    short = [i for i, r in enumerate(raw) if r[0][2] == -1]
    if short:
        embs = enc.embed([audio_clean[int(spans[i][0] * SR):int(spans[i][1] * SR)] for i in short])
        ks = list(centroids)
        Cm = np.stack([centroids[k] for k in ks])
        for i, em in zip(short, embs):
            raw[i] = [(spans[i][0], spans[i][1], int(ks[int(np.argmax(Cm @ em))]))]
    del enc
    _free_gpu()

    # numerotare în ordinea primei intervenții: S1 = primul care vorbește
    order: list[int] = []
    for r in raw:
        for _, _, l in r:
            if l not in order:
                order.append(l)
    names = {l: f"S{i + 1}" for i, l in enumerate(order)}
    out = [[(a, b, names[l]) for a, b, l in r] for r in raw]
    talk: dict[str, float] = {}
    for r in out:
        for a, b, n in r:
            talk[n] = talk.get(n, 0.0) + (b - a)
    return out, {"speakers": len(order), "talk_time_s": {k: round(v, 1) for k, v in talk.items()},
                 "windows": len(windows), "split_segments": sum(len(r) > 1 for r in out),
                 "mode": "fixed" if num_speakers else "auto", "device": _device(), "seconds": round(time.time() - t0, 1)}
