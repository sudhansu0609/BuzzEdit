"""The editor's utterance table: grouping, untranscribed speech, similarity, audio notes."""

import numpy as np

from asr import utterances as ut

SR = ut.SAMPLE_RATE


def _w(text, start, end):
    return {"word": text, "word_native": text, "start": start, "end": end}


def _words(text, start, step=0.4):
    return [_w(t, round(start + i * step, 3), round(start + i * step + step - 0.05, 3))
            for i, t in enumerate(text.split())]


def test_pauses_split_utterances_and_untranscribed_speech_stays_visible():
    words = _words("dosto kya aap", 0.0) + _words("dosto kya aapko pata hai", 3.0)
    speech = [(0.0, 1.2), (1.6, 2.6), (3.0, 5.0)]          # 1.6-2.6: speech, no words
    utts = ut.build(words, speech)
    assert [u.uid for u in utts] == ["U1", "U2", "U3"]
    assert utts[1].text.startswith("[speech, no transcript")
    assert utts[1].word_indices == [] and utts[1].untranscribed_s > 0.9
    assert utts[2].pause_before > 0.3


def test_a_retake_is_pointed_at_even_minutes_away_and_in_another_spelling():
    words = (_words("दोस्तों क्या आपको पता है", 0.0)
             + _words("aaj ki kahani bahut purani hai", 5.0)
             + _words("दोस्तों क्या आपको पता है इंडिया", 120.0))
    utts = ut.build(words)
    first, last = utts[0], utts[-1]
    assert last.uid in [uid for uid, _ in first.similar]
    assert first.uid in last.same_opening
    assert utts[1].uid not in [uid for uid, _ in first.similar]


def _tone(seconds, f0, amplitude=0.3, fade=False):
    t = np.arange(int(seconds * SR)) / SR
    env = np.linspace(1.0, 0.02, t.size) if fade else np.ones(t.size)
    return (amplitude * env * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def _falling(seconds, f0_start, f0_end, amplitude=0.3):
    """A 'finished statement': pitch glides down and the sound decays at the end."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = np.linspace(f0_start, f0_end, t.size)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    env = np.minimum(1.0, (seconds - t) / 0.3)            # 300 ms decay
    return (amplitude * env * np.sin(phase)).astype(np.float32)


def test_audio_notes_hear_a_cut_off_and_a_restart():
    # U1: a level 140 Hz line that stops dead. U2, 0.5 s later, starts at 220 Hz and fades.
    audio = np.zeros(int(4.0 * SR), dtype=np.float32)
    audio[0:int(1.5 * SR)] = _tone(1.5, 140)
    audio[int(2.0 * SR):int(3.5 * SR)] = _tone(1.5, 220, fade=True)
    words = _words("ek do teen", 0.0, step=0.5) + _words("char paanch chhe", 2.0, step=0.5)
    utts = ut.build(words, audio=audio)
    assert "cut off mid-sound" in utts[0].notes
    assert any("higher after an unfinished line" in n for n in utts[1].notes)
    assert "fades out" in utts[1].notes


def test_a_finished_statement_gets_no_alarm():
    # Falling pitch, decaying sound: how a sentence normally ends. Nothing to flag.
    audio = np.zeros(int(3.0 * SR), dtype=np.float32)
    audio[0:int(2.0 * SR)] = _falling(2.0, 180, 120)
    utts = ut.build(_words("yeh kahani yahin khatam", 0.0, step=0.45), audio=audio)
    assert not any(n in utts[0].notes for n in ("cut off mid-sound", "ends on a level pitch",
                                                "ends on a rising pitch"))


def test_the_table_carries_notes_and_pointers():
    words = _words("dosto kya aap", 0.0) + _words("dosto kya aapko pata hai", 3.0)
    text = ut.table(ut.build(words))
    assert text.splitlines()[0].startswith("U1 [0.0-")
    assert "similar to U2" in text


def test_studio_narration_keeps_only_what_is_said():
    script = ("**VISUAL:** [Pitch black.]\n\n**V.O. (Narrator):** (slow) Raat ke sawa do baje... "
              "ek security guard. (pause) Uske haath kaanp rahe hain.\n\n**SOUND:** [drone]")
    spoken = ut.spoken_script(script)
    assert "Pitch black" not in spoken and "drone" not in spoken
    assert "Raat ke sawa do baje" in spoken and "(pause)" not in spoken
