"""A restart from the top is confirmed by its opening said again, not by a greeting-like line alone."""

from asr.restart import confirms_restart, is_greeting, restart_from_top
from asr.utterances import Utterance

GREETING = "Hello doston mera naam hai Sudhansu aur aap dekh rahe hain Raat 3 Baje"
SIGN_OFF = "toh mera naam hai Sudhansu aur aap dekh rahe the Raat 3 Baje"
OPENING = ["aaj ki kahani ek purani haveli ki hai",
           "jahan raat ko koi nahi jaata",
           "gaon wale kehte hain wahan ek aurat rehti hai"]


def _recording(lines):
    """(utterances, words) for (start, text) lines, a word every 0.3 s."""
    utts, words = [], []
    for n, (start, text) in enumerate(lines, 1):
        first = len(words)
        words += [{"word": token, "word_native": token, "start": start + 0.3 * j, "end": start + 0.3 * j + 0.25}
                  for j, token in enumerate(text.split())]
        utts.append(Utterance(f"U{n}", start=start, end=words[-1]["end"],
                              word_indices=list(range(first, len(words))), text=text, pause_before=0.5))
    return utts, words


def test_the_restart_is_the_greeting_whose_opening_is_said_again():
    """Raat3Baje ep2 (2026-10-09): greeting-like lines at 0.4 s, 99.8 s and 515.7 s. Taking the
    last one cut 141.7 s the creator kept; his restart was at 99.8 s, the one greeting followed
    by the opening said again (reworded, respelled). The one at 515.7 s is followed by story."""
    utts, words = _recording([
        (0.4, GREETING), (6.0, OPENING[0]), (10.0, OPENING[1]), (14.0, OPENING[2]),
        (99.8, GREETING),
        (106.0, "aaj ki kahaani ek puraani haveli ke baare mein hai"),
        (110.0, "jahaan raat ko koi bhi nahin jaata"),
        (114.0, "gaon waale kehte hain ki wahaan ek aurat rehti hai"),
        (515.7, GREETING),
        (520.0, "Vikram ne darwaza khola aur andar chala gaya"),
        (524.0, "seedhiyon par kisi ke kadmon ki awaaz aayi"),
        (528.0, "usne torch jalayi to deewar par khoon tha"),
    ])
    cut = restart_from_top(utts, 1400.0, words)
    assert set(cut) == {"U1", "U2", "U3", "U4"}                  # the first attempt, before 99.8 s
    assert cut["U1"].startswith("restart from the top at U5 (100s)") and "after U1 (0s)" in cut["U1"]
    # the greeting at 515.7 s alone removes nothing: what follows it is not the opening
    assert restart_from_top(utts[:4] + utts[8:], 1400.0, words) == {}


def test_a_second_greeting_followed_by_the_same_intro_cuts_everything_before_it():
    """Without words the lines are compared as text. Two opening lines said again confirm a
    restart; one does when the greeting line is said again too, and not on its own."""
    utts, _ = _recording([
        (0.0, GREETING), (5.0, OPENING[0]), (9.0, OPENING[1]), (14.0, "arre ruko ruko phir se karte hain"),
        (20.0, GREETING), (25.0, OPENING[0]), (29.0, OPENING[1]), (34.0, OPENING[2]),
        (300.0, SIGN_OFF),
    ])
    cut = restart_from_top(utts, 320.0)
    assert set(cut) == {"U1", "U2", "U3", "U4"}
    assert cut["U4"].startswith("restart from the top at U5 (20s)")
    assert confirms_restart([GREETING, OPENING[0]], [GREETING, OPENING[0]])
    # a one-line opening that says something else is not a restart
    assert not confirms_restart([GREETING, OPENING[0]], [GREETING, "ek bilkul alag kahani jo kal hui thi"])


def test_a_sign_off_in_the_second_half_never_counts():
    """The sign-off says the greeting's words. Past the first half it is never a restart, even
    when what follows it goes over the opening again."""
    utts, words = _recording([
        (0.0, GREETING), (5.0, OPENING[0]), (9.0, OPENING[1]), (14.0, OPENING[2]),
        (200.0, "aur us raat ke baad haveli ka darwaza phir kabhi nahi khula"),
        (400.0, SIGN_OFF), (405.0, OPENING[0]), (409.0, OPENING[1]), (413.0, OPENING[2]),
    ])
    assert is_greeting(SIGN_OFF)
    assert restart_from_top(utts, 420.0, words) == {} and restart_from_top(utts, 420.0) == {}
    assert set(restart_from_top(utts, 900.0, words)) == {"U1", "U2", "U3", "U4", "U5"}   # only the window


def test_naam_and_dekh_rahe_twenty_words_apart_is_not_a_greeting():
    """A story line may say "mera naam hai" and, much later, "dekh rahe". The greeting says
    both within 12 words of each other, in either script; "naam" alone is not a name phrase."""
    far = ("mera naam hai aur phir woh log us purani haveli ke andar gaye jahan kisi ne bahut "
           "saalon tum mujhe dekh rahe ho")
    assert far.split().index("dekh") == 20
    assert not is_greeting(far)
    assert is_greeting(GREETING) and is_greeting(SIGN_OFF)
    assert is_greeting("नमस्ते दोस्तों, मेरा नाम है सुधांशु और आप देख रहे हैं रात 3 बजे")
    assert not is_greeting("mere saath Vikram naam ka ek purana ranger tha aur sab dekh rahe the")
