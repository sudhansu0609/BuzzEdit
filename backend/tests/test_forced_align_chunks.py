"""The aligner's emission is computed in 30 s chunks so memory stays flat on long takes.

A stand-in model whose frames depend only on their own 25 ms of audio makes any seam or
off-by-one visible: chunked and single-pass output must then be identical frame for frame.
"""

import pytest

torch = pytest.importorskip("torch")

from asr import forced_align as fa  # noqa: E402


class _LocalModel:
    """wav2vec2's frame arithmetic (400-sample window, 320 stride), with a frame-local output."""

    def __call__(self, waveform):
        x = waveform[0]
        frames = (x.numel() - 400) // fa._STRIDE + 1
        windows = x.unfold(0, 400, fa._STRIDE)[:frames]
        out = torch.stack([windows.mean(1), windows.std(1), windows.abs().max(1).values], dim=1)
        return out.unsqueeze(0), None


def test_chunked_emission_matches_a_single_pass():
    torch.manual_seed(0)
    waveform = torch.randn(1, 95 * fa._SAMPLE_RATE)               # 95 s: three full chunks and a tail
    model = _LocalModel()
    whole, _ = model(waveform)
    chunked = fa._emission(model, waveform, "cpu")
    assert abs(chunked.size(1) - whole.size(1)) <= 1
    n = min(chunked.size(1), whole.size(1))
    assert torch.allclose(chunked[:, :n], whole[:, :n])


def test_a_short_recording_is_one_pass():
    waveform = torch.randn(1, 20 * fa._SAMPLE_RATE)
    model = _LocalModel()
    assert torch.equal(fa._emission(model, waveform, "cpu"), model(waveform)[0])
