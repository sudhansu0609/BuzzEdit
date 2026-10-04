"""Ground truth from a hand edit: which parts of the raw recording the editor kept.

    ./.venv/Scripts/python.exe backend/tools/autocut_truth.py <raw-video> <edited-video> <name>

The user cuts a recording by hand (Premiere, BuzzEdit, anything) and exports it.
This recovers, from the two files alone, exactly which source spans the export
plays and in what order, by matching the export's audio back onto the raw
recording's audio. It writes `data/eval/<name>.json` in the format
`autocut_eval.py` scores against — the removed source spans (`spans`, each
`silence` or `speech`) — plus the kept ones (`kept`, with where each sits in the
export), so every hand edit the user makes becomes a test the auto-edit can be
held to.

How the matching works, in three steps:

1. **Find candidates.** Both soundtracks become 10ms log-mel frames, normalised per
   band so the export's EQ and mastering matter less, and every 0.1s of the export
   is searched for across the whole recording (an FFT correlation). That nominates
   a few candidate source offsets per window — the true one among them, plus
   look-alikes.
2. **Verify on the waveform.** Spectra cannot tell two takes of the same sentence
   apart reliably: an export that was denoised and EQ'd matches its own source
   take at a cosine of only ~0.7, and another take of the same words comes close.
   The waveform can. An export is the source's samples run through filters, so at
   the exact sample offset the two still correlate — after pre-emphasis, ~0.8 on
   speech even for a denoised, EQ'd export — while another take (different samples)
   stays near 0. Each candidate is refined to the sample and kept only if it does.
3. **Explain the export.** A Viterbi pass assigns every 10ms of the export to one
   verified offset, paying a fixed penalty per switch, so the answer is the fewest
   edit points that account for the audio. Searching the whole recording means a
   cut back to an earlier take is found as easily as a cut forward.

Silence carries little evidence (a denoised export's pauses are near-digital-zero),
with one exception: export silence over *loud speech* in the source means that speech
was not kept, since no denoiser removes it. Otherwise an edit point inside a pause is
placed at the middle of it: the export only fixes how long the pause is, not which
side of the cut it came from.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 16000
HOP = 160                         # 10 ms analysis frames
FPS = SR / HOP
N_MELS = 64

SEARCH_WINDOW_FRAMES = 100        # 1 s of export audio per position search
SEARCH_HOP_FRAMES = 10
CANDIDATES_PER_WINDOW = 3
NMS_FRAMES = 30                   # peaks closer than this are one candidate
MIN_SEARCH_SPEECH = 25.0          # a window with less speech than this is not searched
MIN_CANDIDATE_SCORE = 0.35

# Waveform correlation runs on pre-emphasised audio. Raw speech is dominated by its
# low harmonics, and any two voiced stretches correlate at *some* lag, so a random
# position reaches 0.2 in a 1 s window. Flattening the spectrum first leaves only
# the identical samples correlating. Measured on a denoised, EQ'd export
# (Raat3Baje ep1, 2026-10-04):
#   1 s windows:  true take 0.78-0.84, random max-over-lags <= 0.13
#   40 ms blocks: true median 0.81 (p10 0.59), random p99.9 0.39
PRE_EMPHASIS = 0.97
VERIFY_LAG = 2 * HOP              # sample search either side of a candidate's frame offset
VERIFY_NCC = 0.30                 # 1 s waveform correlation a real match clears
SAME_OFFSET_SAMPLES = 40          # verified offsets this close are one stretch
BLOCK = 4 * HOP                   # 40 ms correlation block centred on each 10 ms frame

MATCH_NCC = 0.30                  # block correlation above this: the offset explains the frame
UNMATCHED_NCC = 0.15              # an "inserted material" state scores as this
CUT_PENALTY = 4.0                 # cost of one edit point, in frame-evidence units
SILENCE_WEIGHT = 0.02             # a denoised pause carries no evidence of its own...
SILENT_OVER_SPEECH = 0.3          # ...but silence where the source has loud speech is
                                  # evidence against the offset: no denoiser removes that
MIN_SILENT_RUN = 3                # frames; an edit point inside a pause this long is centred
MIN_SPEECH_IN_REMOVED_S = 0.25    # a removed span with less speech is "silence"


# --- audio and features ------------------------------------------------------

def load_audio(path: str) -> np.ndarray:
    """Mono 16 kHz float32, decoded by ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def _device(name: Optional[str] = None):
    import torch
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_mel(audio: np.ndarray, device) -> Tuple[Any, np.ndarray]:
    """(frames x N_MELS log-mel tensor on `device`, per-frame energy in dB)."""
    import torch
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=512, win_length=400, hop_length=HOP, n_mels=N_MELS).to(device)
    with torch.no_grad():
        spec = mel(torch.from_numpy(audio).to(device))          # [N_MELS, T]
        energy_db = 10.0 * torch.log10(spec.sum(0) + 1e-10)
        feats = torch.log(spec + 1e-8).T.contiguous()            # [T, N_MELS]
    return feats, energy_db.cpu().numpy()


def speechness(energy_db: np.ndarray) -> np.ndarray:
    """0..1 per frame: how clearly the frame is speech rather than room tone.
    Thresholded between this file's own noise floor and speech peak, as asr/vad.py does."""
    floor = float(np.percentile(energy_db, 10))
    peak = float(np.percentile(energy_db, 95))
    threshold = floor + 0.3 * (peak - floor)
    return np.clip((energy_db - threshold) / 6.0, 0.0, 1.0).astype(np.float32)


def normalise(feats, speech: np.ndarray):
    """Per-band mean/variance over the file's speech, then unit length per frame."""
    import torch
    mask = torch.from_numpy(speech > 0.5).to(feats.device)
    voiced = feats[mask] if int(mask.sum()) > 100 else feats
    out = (feats - voiced.mean(0)) / (voiced.std(0) + 1e-6)
    return out / (out.norm(dim=1, keepdim=True) + 1e-6)


# --- step 1: candidates from the spectra ---------------------------------------

VERIFY_WINDOWS = 3                # each candidate is checked on its most speech-rich windows


def candidate_offsets(cut_n, src_n, cut_speech: np.ndarray) -> Dict[int, List[Tuple[float, int]]]:
    """{source-minus-export frame offset: [(evidence, export frame), ...]} — for each
    offset, the windows that nominated it, best evidence first (at most VERIFY_WINDOWS).

    Each 1 s window of the export (speech-weighted) is correlated against the whole
    recording; the top few peaks become candidates. A kept stretch nominates its own
    offset from every window inside it, so a true offset is not missed. Evidence is the
    score times how much of the window is speech: the window that verifies an offset
    must be one with speech in it — a long kept stretch once lost all 50 s of itself
    because its single best-scoring window was two-thirds pause.
    """
    import torch
    t_src, t_cut = src_n.shape[0], cut_n.shape[0]
    w = SEARCH_WINDOW_FRAMES
    nfft = 1 << int(np.ceil(np.log2(t_src + w)))
    src_f = torch.fft.rfft(src_n, n=nfft, dim=0)                 # [nfft/2+1, N_MELS]
    weights = torch.from_numpy(cut_speech).to(cut_n.device)
    found: Dict[int, List[Tuple[float, int]]] = {}
    for t in range(0, max(1, t_cut - w), SEARCH_HOP_FRAMES):
        win_w = weights[t:t + w]
        total = float(win_w.sum())
        if total < MIN_SEARCH_SPEECH:
            continue
        window = (cut_n[t:t + w] * win_w[:, None]).flip(0)
        prod = (src_f * torch.fft.rfft(window, n=nfft, dim=0)).sum(1)
        conv = torch.fft.irfft(prod, n=nfft)
        scores = conv[w - 1: w - 1 + t_src - w + 1] / total     # scores[p]: window at source frame p
        for _ in range(CANDIDATES_PER_WINDOW):
            p = int(torch.argmax(scores))
            value = float(scores[p])
            if value < MIN_CANDIDATE_SCORE:
                break
            windows = found.setdefault(p - t, [])
            windows.append((value * total / w, t))
            windows.sort(reverse=True)
            del windows[VERIFY_WINDOWS:]
            scores[max(0, p - NMS_FRAMES): p + NMS_FRAMES + 1] = -1.0
    return found


# --- step 2: verification on the waveform --------------------------------------

def pre_emphasis(audio: np.ndarray) -> np.ndarray:
    out = np.empty_like(audio)
    out[:1] = audio[:1]
    out[1:] = audio[1:] - PRE_EMPHASIS * audio[:-1]
    return out


def _padded(audio: np.ndarray, pad: int, device):
    import torch
    return torch.nn.functional.pad(torch.from_numpy(audio).to(device), (pad, pad))


def verify_candidates(cut_t, src_p, pad: int,
                      candidates: Dict[int, List[Tuple[float, int]]]) -> Dict[int, float]:
    """{sample offset (source sample = export sample + offset): waveform correlation}
    for the candidates the waveform confirms, each refined to the exact sample on the
    best of its nominating windows."""
    import torch
    n = SEARCH_WINDOW_FRAMES * HOP
    lags = 2 * VERIFY_LAG + 1
    verified: Dict[int, float] = {}
    for offset, windows in candidates.items():
        best_value, best_exact = -1.0, None
        for _evidence, t in windows:
            a = t * HOP
            c = cut_t[a:a + n]
            if c.numel() < n:
                continue
            c_norm = float(torch.sqrt((c * c).sum()))
            if c_norm < 1e-6:
                continue
            base = pad + a + offset * HOP - VERIFY_LAG
            if base < 0 or base + n + lags > src_p.numel():
                continue
            s = src_p[base: base + n + lags - 1]
            dots = s.unfold(0, n, 1) @ c                              # [lags]
            cs = torch.cumsum(torch.nn.functional.pad(s * s, (1, 0)), 0)
            energy = cs[n:n + lags] - cs[:lags]
            ncc = dots / (c_norm * torch.sqrt(energy.clamp_min(0)) + 1e-9)
            k = int(torch.argmax(ncc))
            if float(ncc[k]) > best_value:
                best_value, best_exact = float(ncc[k]), offset * HOP + k - VERIFY_LAG
        if best_exact is not None and best_value >= VERIFY_NCC:
            if best_value > verified.get(best_exact, -1.0):
                verified[best_exact] = best_value
    return verified


def merge_offsets(verified: Dict[int, float], tolerance: int = SAME_OFFSET_SAMPLES) -> List[int]:
    """Verified offsets within `tolerance` samples of a stronger one are the same stretch."""
    kept: List[int] = []
    for offset, _value in sorted(verified.items(), key=lambda kv: -kv[1]):
        if all(abs(offset - k) > tolerance for k in kept):
            kept.append(offset)
    return sorted(kept)


def block_ncc(cut_blocks, cut_energy, src_p, pad: int, offsets: List[int], t_cut: int):
    """[len(offsets), t_cut]: waveform correlation of each export frame's 20 ms block
    with the source block each offset maps it to."""
    import torch
    out = torch.zeros((len(offsets), t_cut), device=cut_blocks.device)
    span = (t_cut - 1) * HOP + BLOCK
    for i, d in enumerate(offsets):
        start = pad + d - BLOCK // 2                 # blocks are centred on their frame
        s = src_p[start: start + span]
        if s.numel() < span:
            continue
        sb = s.unfold(0, BLOCK, HOP)[:t_cut]
        out[i] = (cut_blocks * sb).sum(1) / torch.sqrt(cut_energy * (sb * sb).sum(1) + 1e-12)
    return out


# --- step 3: explaining the export --------------------------------------------

def viterbi(emission: np.ndarray, penalty: float) -> np.ndarray:
    """Best piecewise-constant state path through `emission` [T, S]; switching
    state costs `penalty`. Returns the state index per frame."""
    t_len, n_states = emission.shape
    score = emission[0].astype(np.float64).copy()
    jumped = np.zeros((t_len, n_states), dtype=bool)
    best_prev = np.zeros(t_len, dtype=np.int32)
    for t in range(1, t_len):
        b = int(np.argmax(score))
        jump = score[b] - penalty
        take = jump > score
        score = np.where(take, jump, score) + emission[t]
        jumped[t] = take
        best_prev[t] = b
    path = np.empty(t_len, dtype=np.int32)
    s = int(np.argmax(score))
    for t in range(t_len - 1, -1, -1):
        path[t] = s
        if t and jumped[t, s]:
            s = int(best_prev[t])
    return path


def centre_cuts_in_pauses(path: np.ndarray, speech: np.ndarray, loud_under) -> np.ndarray:
    """Move each switch that lands in a pause towards the middle of that pause: silence
    carries no evidence, so the Viterbi's choice inside one is arbitrary.

    `loud_under(state, frames)` says, per export frame, whether the source frame that state
    maps it to is speech. A pause in the export over source speech means that speech was
    cut, so the move never hands such a frame to that state — without this a silent title
    card at the head of an export dragged the first kept span 0.8 s into the speech before it.
    """
    path = path.copy()
    voiced = speech > 0.5
    switches = np.nonzero(path[1:] != path[:-1])[0] + 1
    for b in switches:
        if voiced[b] and voiced[b - 1]:
            continue                                    # cut through speech: evidence placed it
        lo = b
        while lo > 0 and not voiced[lo - 1]:
            lo -= 1
        hi = b
        while hi < len(path) and not voiced[hi]:
            hi += 1
        if hi - lo < MIN_SILENT_RUN:
            continue
        before, after = path[b - 1], path[b]
        # Only move within the run the two neighbouring states share.
        if path[lo] != before and lo < b:
            continue
        if hi - 1 >= b and path[hi - 1] != after:
            continue
        frames = np.arange(lo, hi)
        loud_after = np.nonzero(loud_under(after, frames))[0]
        loud_before = np.nonzero(loud_under(before, frames))[0]
        earliest = lo + (loud_after[-1] + 1 if loud_after.size else 0)
        latest = lo + (loud_before[0] if loud_before.size else hi - lo)
        if earliest > latest:
            continue
        split = min(max((lo + hi) // 2, earliest), latest)
        path[lo:split] = before
        path[split:hi] = after
    return path


def align_edit(src_audio: np.ndarray, cut_audio: np.ndarray, device=None,
               log=print) -> Dict[str, Any]:
    """Which source spans the export plays, in export order."""
    import torch
    device = device or _device()
    t0 = time.time()
    with torch.no_grad():
        src_feats, src_db = log_mel(src_audio, device)
        cut_feats, cut_db = log_mel(cut_audio, device)
        src_speech, cut_speech = speechness(src_db), speechness(cut_db)
        src_n = normalise(src_feats, src_speech)
        cut_n = normalise(cut_feats, cut_speech)
        del src_feats, cut_feats
        t_cut = cut_n.shape[0]

        candidates = candidate_offsets(cut_n, src_n, cut_speech)
        del src_n, cut_n
        log(f"  {len(candidates)} spectral candidates ({time.time() - t0:.1f}s)")

        pad = len(cut_audio) + 4 * BLOCK
        src_p = _padded(pre_emphasis(src_audio), pad, device)
        cut_t = torch.from_numpy(pre_emphasis(cut_audio)).to(device)
        verified = verify_candidates(cut_t, src_p, pad, candidates)
        offsets = merge_offsets(verified)
        log(f"  {len(verified)} confirmed by the waveform -> {len(offsets)} offsets "
            f"({time.time() - t0:.1f}s)")

        cut_p = torch.nn.functional.pad(cut_t, (BLOCK // 2, BLOCK))
        cut_blocks = cut_p.unfold(0, BLOCK, HOP)[:t_cut]
        cut_energy = (cut_blocks * cut_blocks).sum(1)
        ncc = block_ncc(cut_blocks, cut_energy, src_p, pad, offsets, t_cut)

        speech_t = torch.from_numpy(cut_speech).to(device)
        weight = SILENCE_WEIGHT + (1.0 - SILENCE_WEIGHT) * speech_t
        emission = weight[None, :] * (ncc - MATCH_NCC)
        # Silence in the export over loud speech in the source: that speech was not kept.
        src_speech_t = torch.from_numpy(src_speech).to(device)
        frames = torch.arange(t_cut, device=device)
        for i, d in enumerate(offsets):
            mapped = frames + int(round(d / HOP))
            inside = (mapped >= 0) & (mapped < src_speech_t.numel())
            loud = torch.zeros(t_cut, device=device)
            loud[inside] = src_speech_t[mapped[inside]]
            emission[i] -= SILENT_OVER_SPEECH * (1.0 - speech_t) * loud
        unmatched = weight[None, :] * (UNMATCHED_NCC - MATCH_NCC)
        emission = torch.cat([emission, unmatched], 0).T.contiguous().cpu().numpy()
        ncc_np = ncc.cpu().numpy()
    src_voiced = src_speech > 0.5

    def loud_under(state: int, frames: np.ndarray) -> np.ndarray:
        if state >= len(offsets):                       # inserted material maps nowhere
            return np.zeros(frames.size, dtype=bool)
        mapped = frames + int(round(offsets[state] / HOP))
        inside = (mapped >= 0) & (mapped < src_voiced.size)
        out = np.zeros(frames.size, dtype=bool)
        out[inside] = src_voiced[mapped[inside]]
        return out

    path = centre_cuts_in_pauses(viterbi(emission, CUT_PENALTY), cut_speech, loud_under)
    log(f"  explained ({time.time() - t0:.1f}s)")

    unmatched_state = len(offsets)
    segments: List[Dict[str, Any]] = []
    start = 0
    for t in range(1, len(path) + 1):
        if t < len(path) and path[t] == path[start]:
            continue
        state = int(path[start])
        voiced = cut_speech[start:t] > 0.5
        seg: Dict[str, Any] = {
            "cut_start": round(start / FPS, 3),
            "cut_end": round(t / FPS, 3),
            "speech_s": round(float(voiced.sum()) / FPS, 2),
        }
        if state == unmatched_state:
            seg["unmatched"] = True
        else:
            d = offsets[state]
            seg["src_start"] = round((start * HOP + d) / SR, 3)
            seg["src_end"] = round((t * HOP + d) / SR, 3)
            seg["offset_samples"] = int(d)
            if voiced.any():
                own = ncc_np[state, start:t][voiced]
                others = np.delete(ncc_np[:, start:t][:, voiced], state, axis=0)
                rival = float(others.mean(1).max()) if others.size else 0.0
                seg["match"] = round(float(own.mean()), 3)
                seg["margin"] = round(float(own.mean()) - rival, 3)
        segments.append(seg)
        start = t
    return {"segments": segments, "src_duration": round(len(src_audio) / SR, 3),
            "cut_duration": round(len(cut_audio) / SR, 3)}


# --- from kept spans to ground truth --------------------------------------------

def removed_spans(kept: List[Tuple[float, float]], src_duration: float,
                  min_gap: float = 0.05) -> List[Tuple[float, float]]:
    """The source time no kept span plays."""
    spans: List[Tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(kept):
        if start - cursor >= min_gap:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    if src_duration - cursor >= min_gap:
        spans.append((cursor, src_duration))
    return spans


def classify(spans: List[Tuple[float, float]], speech_map) -> List[Dict[str, Any]]:
    out = []
    for start, end in spans:
        speech_s = speech_map.speech_fraction(start, end) * (end - start) if speech_map else 0.0
        out.append({"start": round(start, 3), "end": round(end, 3),
                    "kind": "speech" if speech_s >= MIN_SPEECH_IN_REMOVED_S else "silence",
                    "speech_s": round(speech_s, 2)})
    return out


def build_truth(raw: str, edited: str, work_dir: Path, log=print) -> Dict[str, Any]:
    import soundfile as sf
    from asr.vad import analyse_speech

    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / (Path(raw).stem + "_16k.wav")
    if wav.exists():
        src_audio, _ = sf.read(str(wav), dtype="float32")
    else:
        log(f"decoding {raw}")
        src_audio = load_audio(raw)
        sf.write(str(wav), src_audio, SR)
    log(f"decoding {edited}")
    cut_audio = load_audio(edited)
    log("matching the edit onto the recording")
    aligned = align_edit(src_audio, cut_audio, log=log)
    speech_map = analyse_speech(str(wav))

    matched = [s for s in aligned["segments"] if not s.get("unmatched")]
    kept = [(s["src_start"], s["src_end"]) for s in matched]
    spans = classify(removed_spans(kept, aligned["src_duration"]), speech_map)
    reordered = sum(1 for a, b in zip(matched, matched[1:]) if b["src_start"] < a["src_end"] - 0.5)
    return {
        "source": str(raw),
        "edited": str(edited),
        "src_duration": aligned["src_duration"],
        "cut_duration": aligned["cut_duration"],
        "kept": aligned["segments"],
        "spans": spans,
        "stats": {
            "kept_segments": len(matched),
            "kept_s": round(sum(e - s for s, e in kept), 2),
            "unmatched_s": round(sum(s["cut_end"] - s["cut_start"]
                                     for s in aligned["segments"] if s.get("unmatched")), 2),
            "removed_speech_s": round(sum(s["end"] - s["start"] for s in spans if s["kind"] == "speech"), 2),
            "removed_silence_s": round(sum(s["end"] - s["start"] for s in spans if s["kind"] == "silence"), 2),
            "reordered_segments": reordered,
            "weak_segments": sum(1 for s in matched if s.get("margin", 1.0) < 0.1 and s["speech_s"] > 0.3),
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw", help="the unedited recording")
    p.add_argument("edited", help="the user's own cut of it")
    p.add_argument("name", help="ground-truth file stem; written to data/eval/<name>.json")
    p.add_argument("--work-dir", default=None, help="where the decoded source WAV is cached")
    args = p.parse_args(argv)

    from config import DATA_DIR
    eval_dir = DATA_DIR / "eval"
    work_dir = Path(args.work_dir) if args.work_dir else eval_dir / "cache"
    truth = build_truth(args.raw, args.edited, work_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    out = eval_dir / f"{args.name}.json"
    out.write_text(json.dumps(truth, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(truth["stats"], indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
