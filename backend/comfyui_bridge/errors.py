"""Exceptions the ComfyUI bridge raises that must end a job rather than let
it quietly render without the pictures it asked for.

2026-10-20: ComfyUI was down for 8 days and the presentation pass kept
"degrading" instead of failing -- 263 beats planned, 0 assets generated, 0
B-roll placed, delivered anyway. `ComfyUIUnavailableError` is how a pass says
"I waited, it never came back, and the cache does not cover enough of the
plan" so the job ends failed and gets retried later instead.
"""


class ComfyUIUnavailableError(RuntimeError):
    """ComfyUI did not come back within the wait budget, and cached/stock
    assets do not cover enough of what the plan wanted. Raised instead of
    letting a presentation pass render with (effectively) no B-roll."""
