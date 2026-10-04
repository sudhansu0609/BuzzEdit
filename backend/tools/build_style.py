"""Turn the creator's hand cuts into the AI editor's stored style examples.

    ./.venv/Scripts/python.exe backend/tools/build_style.py life3baje_ep1 raat3baje_ep1

Each name is an answer key in `data/eval/<name>.json` (tools/autocut_truth.py). Its recording is
prepared exactly as the bench prepares it (cached transcript, VAD, audio notes), the excerpts
asr.style picks are stored in `data/style/examples.jsonl` as "hand_cut" examples with the key's
channel (the name before the first underscore), replacing that key's earlier excerpts. The
planner then reads them for every new recording (asr.style.stored_style), never from the
recording being cut when that recording is itself the key (`exclude_source`).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("keys", nargs="+")
    args = p.parse_args(argv)

    from asr import style
    from tools.editor_bench import prepare

    prep_args = argparse.Namespace(script="", no_notes=False, no_hints=False)

    async def run():
        for name in args.keys:
            prepared = await prepare(name, prep_args)
            for u in prepared["utterances"]:     # the product cuts before any script arrives
                u.script = None
            cut = [(float(s["start"]), float(s["end"])) for s in prepared["key"]["spans"]]
            decided = style.labels(prepared["utterances"], cut,
                                   [tuple(e) for e in prepared["key"].get("editorial", [])])
            texts = style.excerpts(prepared["utterances"], decided, name)
            channel = name.split("_")[0]
            total = style.add_examples([{"kind": "hand_cut", "channel": channel, "source": name, "text": t}
                                        for t in texts], replace_source=name)
            print(json.dumps({"key": name, "channel": channel, "excerpts": len(texts), "stored": total}))
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
