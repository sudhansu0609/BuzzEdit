"""Child-process creation flags.

The desktop app launches with no console of its own (see START_APP.vbs), so a
console-subsystem child — ffmpeg, ffprobe, the LM server — allocates a brand new
console window when it starts. During a render that is one black window flashing
open and shut per clip, on top of the app. CREATE_NO_WINDOW suppresses it while
still leaving stdout/stderr pipes usable, which DETACHED_PROCESS would not.

Pass `creationflags=NO_WINDOW` to every subprocess spawn in the backend. The
value is 0 off Windows, where the flag does not exist and is simply a no-op.
"""

import subprocess
import sys

NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
