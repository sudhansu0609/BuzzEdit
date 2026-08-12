from backend.asr.disfluency import analyze_disfluencies

def test_analyze_disfluencies_fillers_and_repeats():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "um", "start": 0.5, "end": 0.8},
        {"word": "we", "start": 0.8, "end": 1.0},
        {"word": "we", "start": 1.0, "end": 1.2},
        {"word": "are", "start": 1.2, "end": 1.4},
        {"word": "here", "start": 1.4, "end": 1.6},
    ]

    annotated = analyze_disfluencies(words)

    assert annotated[0]["disfluency"] is False  # Hello
    assert annotated[1]["disfluency"] is True   # um (filler)
    assert annotated[2]["disfluency"] is False  # we (first)
    assert annotated[3]["disfluency"] is True   # we (repeat)
    assert annotated[4]["disfluency"] is False  # are
    assert annotated[5]["disfluency"] is False  # here
