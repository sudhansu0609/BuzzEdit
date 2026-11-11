"""Builders for ``FlowDiagram`` and ``IconList`` (MOTION_GRAPHICS_PLAN.md M9).

``[flow: step A ; step B ; step C | layout=chain|cycle|branch | seconds=N]``
-> a ``flow_diagram`` beat -> ``build_flow_diagram`` -> a FlowDiagram job:
2-6 rounded boxes connected by arrows that draw in order, `chain` left to
right (wrapping to two rows above 4), `cycle` on a circle closing the loop,
`branch` fanning out from the first box.

``[icons: label ; label ; ... | mode=list|pros_cons | seconds=N]`` -> an
``icon_list`` beat -> ``build_icon_list`` -> an IconList job: 2-6 labels each
given a bundled Lucide icon by keyword (``icon_for_label``), or split into a
pros/cons grid by a leading ``+``/``-`` on the label.

Both (presentation/script.py::directive_beats) -> this module
(registered in ``presentation/motion.py::BUILDERS``). A render the engine
cannot make falls back to an existing PIL card -- the closest fit
graphics.py already has to "boxes in order" / "labels in a row" is a
vertical numbered-point list (``render_flow_fallback``) and a stack of pills
(``render_icon_list_fallback``), used by ``motion.render_fallback``.
"""

import re
from typing import Any, Dict, List, Optional

from .models import Beat
from .motion import MotionContext, _beat_duration

_LAYOUTS = ("chain", "cycle", "branch")
_MODES = ("list", "pros_cons")

# Label keyword -> Lucide icon slug (lucide-react's own kebab-case naming;
# every slug here is verified against the pinned lucide-react 1.56.0 package
# -- motion/node_modules/lucide-react -- via its `icons` map, PascalCase).
# Checked in order, first match wins; a label matching nothing gets the
# neutral default. ~120 entries (MOTION_GRAPHICS_PLAN.md M9b review): a label
# is tokenised into words and each word is matched *whole*, with simple
# stemming (plural/verb suffixes) against a keyword -- never a substring
# match against the raw label text, which used to let "food"'s "eat" keyword
# fire on "Great support" (the "eat" hiding inside "gr-EAT").
_ICON_KEYWORDS: List[tuple] = [
    (("money", "cost", "price", "expensive", "cheap", "budget", "finance", "afford", "affordable"), "wallet"),
    (("coin", "coins", "currency", "cash"), "coins"),
    (("payment", "pay", "checkout", "credit", "card"), "credit-card"),
    (("savings", "save", "saving", "piggy"), "piggy-bank"),
    (("bank", "banking", "loan", "lending"), "landmark"),
    (("slow", "slower", "longer", "wait", "waiting", "delay", "delayed"), "hourglass"),
    (("time", "clock", "hour", "minute", "deadline", "late"), "clock"),
    (("fast", "quick", "speed", "rapid", "instant"), "zap"),
    (("water", "sea", "ocean", "river", "flood", "wave", "waves"), "waves-horizontal"),
    (("danger", "risk", "warning", "hazard", "threat", "unsafe"), "triangle-alert"),
    (("safe", "safety", "security", "secure", "protect", "protection", "protected"), "shield-check"),
    (("privacy", "secret", "confidential", "password", "lock", "locked"), "lock"),
    (("food", "eat", "meal", "restaurant", "cuisine", "hungry", "snack"), "utensils"),
    (("world", "global", "international", "earth", "everywhere", "worldwide"), "globe"),
    (("people", "crowd", "public", "community", "audience", "team", "teamwork"), "users"),
    (("person", "individual", "user", "profile", "member"), "user"),
    (("home", "house", "family", "household"), "house"),
    (("ship", "boat", "sail", "voyage", "vessel"), "ship"),
    (("phone", "mobile", "call", "smartphone", "cellphone"), "smartphone"),
    (("health", "medical", "doctor", "hospital", "illness", "sick", "disease"), "heart-pulse"),
    (("medicine", "pill", "drug", "dose", "dosage"), "pill"),
    (("stethoscope", "diagnosis", "checkup"), "stethoscope"),
    (("vaccine", "syringe", "injection"), "syringe"),
    (("idea", "innovation", "insight", "concept", "brainstorm"), "lightbulb"),
    (("location", "place", "map", "city", "destination", "address"), "map-pin"),
    (("growth", "increase", "rise", "trend", "profit", "upward"), "trending-up"),
    (("shrink", "decrease", "decline", "drop", "downturn"), "trending-down"),
    (("school", "education", "student", "learning", "university", "learn", "study"), "graduation-cap"),
    (("book", "reading", "read", "novel", "library"), "book-open"),
    (("write", "writing", "pen", "edit", "draft", "author"), "pen-line"),
    (("work", "job", "career", "office", "business", "corporate"), "briefcase"),
    (("building", "headquarters"), "building"),
    (("love", "emotion", "relationship", "romance"), "heart"),
    (("happy", "satisfied", "positive", "approve", "approved", "like", "agree"), "thumbs-up"),
    (("sad", "unhappy", "dissatisfied", "negative", "reject", "dislike", "disagree"), "thumbs-down"),
    (("email", "letter", "contact", "mail"), "mail"),
    (("chat", "conversation", "talk", "discuss", "discussion", "message"), "message-circle"),
    (("search", "find", "investigate", "research", "lookup"), "search"),
    (("shopping", "buy", "purchase", "market", "sale", "store", "retail", "shop"), "shopping-cart"),
    (("music", "song", "sound"), "music"),
    (("podcast", "microphone", "voice"), "mic"),
    (("headphones", "earbuds"), "headphones"),
    (("video", "film", "movie", "clip", "footage"), "video"),
    (("camera", "photo", "picture", "photograph"), "camera"),
    (("travel", "flight", "airport", "journey", "trip", "vacation"), "plane"),
    (("car", "drive", "vehicle", "traffic", "road"), "car"),
    (("truck", "delivery", "shipping", "logistics", "freight"), "truck"),
    (("fire", "burn", "heat", "flame"), "flame"),
    (("law", "legal", "court", "justice", "lawsuit"), "scale"),
    (("science", "lab", "experiment", "chemistry"), "flask-conical"),
    (("weather", "climate", "rain", "storm"), "cloud"),
    (("sun", "sunny", "bright", "daylight"), "sun"),
    (("night", "dark", "midnight", "darkness"), "moon"),
    (("cold", "ice", "winter", "freeze", "freezing"), "snowflake"),
    (("nature", "forest", "tree", "environment", "green"), "trees"),
    (("recycle", "recycling", "reuse", "sustainable", "sustainability"), "recycle"),
    (("leaf", "natural", "organic"), "leaf"),
    (("plant", "grow", "garden", "seed"), "sprout"),
    (("farm", "agriculture", "crop", "harvest"), "wheat"),
    (("date", "calendar", "schedule", "event", "booking"), "calendar"),
    (("goal", "target", "aim", "objective"), "target"),
    (("quality", "star", "rating", "best", "premium"), "star"),
    (("award", "win", "winning", "prize", "achievement"), "trophy"),
    (("medal", "champion", "winner"), "medal"),
    (("crown", "leader", "leadership"), "crown"),
    (("gem", "valuable", "precious", "diamond"), "gem"),
    (("question", "doubt", "confusion", "unknown", "unclear"), "circle-question-mark"),
    (("tool", "fix", "repair", "build", "construction", "maintenance"), "wrench"),
    (("factory", "industry", "manufacturing", "production"), "factory"),
    (("data", "database", "storage", "server", "backend"), "database"),
    (("chart", "graph", "statistics", "analytics", "metrics", "dashboard"), "chart-column"),
    (("support", "help", "assistance", "assist"), "life-buoy"),
    (("internet", "online", "wifi", "connectivity", "connected"), "wifi"),
    (("bluetooth", "wireless"), "bluetooth"),
    (("energy", "power", "electricity", "electric"), "battery-charging"),
    (("plug", "outlet", "charger", "charging"), "plug"),
    (("death", "dead", "kill", "fatal"), "skull"),
    (("document", "file", "paper", "form", "report"), "file-text"),
    (("folder", "files", "organize", "organization"), "folder"),
    (("upload", "export"), "upload"),
    (("download", "import"), "download"),
    (("share", "sharing", "social"), "share-2"),
    (("link", "connect", "connection", "integration", "integrate"), "link"),
    (("layers", "structure", "complexity", "complex"), "layers"),
    (("package", "inventory", "module", "parcel", "box"), "package"),
    (("computer", "cpu", "processor", "hardware", "processing"), "cpu"),
    (("monitor", "desktop", "screen", "display"), "monitor"),
    (("laptop", "remote", "notebook"), "laptop"),
    (("printer", "print", "printing"), "printer"),
    (("code", "programming", "developer", "software", "coding"), "code"),
    (("bug", "error", "glitch", "problem", "issue"), "bug"),
    (("robot", "ai", "automation", "automated", "artificial"), "bot"),
    (("brain", "mind", "smart", "intelligence", "intelligent"), "brain"),
    (("eye", "see", "view", "watch", "visible", "vision"), "eye"),
    (("ear", "hear", "listen", "hearing"), "ear"),
    (("hand", "touch", "gesture"), "hand"),
    (("settings", "config", "configuration", "setup", "customize"), "settings"),
    (("key", "access", "unlock", "authorize", "authorized"), "key"),
    (("flag", "country", "nation", "national"), "flag"),
    (("compass", "direction", "navigate", "navigation", "guide"), "compass"),
    (("anchor", "stability", "stable", "steady"), "anchor"),
    (("signal", "network", "broadcast"), "signal"),
    (("tv", "television"), "tv"),
    (("newspaper", "news", "media", "press"), "newspaper"),
    (("bell", "notification", "alert", "reminder", "notify"), "bell"),
    (("percent", "percentage", "rate", "ratio", "discount"), "percent"),
    (("gift", "present", "bonus", "reward"), "gift"),
    (("vote", "election", "democracy", "poll"), "vote"),
    (("rocket", "launch", "startup", "liftoff"), "rocket"),
    (("puzzle", "solve", "solution"), "puzzle"),
    (("sparkles", "new", "fresh", "latest", "magic"), "sparkles"),
    (("archive", "old", "legacy", "outdated", "vintage"), "archive"),
    (("handshake", "deal", "partnership", "agreement", "partner"), "handshake"),
    (("accessibility", "inclusive", "accessible"), "accessibility"),
    (("language", "languages", "translate", "translation", "multilingual"), "languages"),
    (("umbrella", "insurance"), "umbrella"),
    (("thermometer", "temperature", "fever"), "thermometer"),
    (("baby", "child", "kids", "children"), "baby"),
    (("calculator", "math", "compute", "computation"), "calculator"),
    (("ruler", "measure", "measurement", "precise", "precision"), "ruler"),
    (("palette", "design", "art", "artistic"), "palette"),
    (("gamepad", "gaming", "game", "play"), "gamepad-2"),
]
_DEFAULT_ICON = "circle-dot"

_WORD_RE = re.compile(r"[a-z]+")
# Simple stems a label's own word may carry over a bare keyword -- "works"
# still reads as "work", "cheaper" still reads as "cheap" -- without ever
# matching a keyword that merely appears *inside* a longer, unrelated word.
_STEM_SUFFIXES = ("ing", "es", "ed", "er", "est", "ly", "s")


def _word_matches(word: str, keyword: str) -> bool:
    if word == keyword:
        return True
    return any(word == keyword + suffix for suffix in _STEM_SUFFIXES)


def icon_for_label(label: str) -> str:
    """A label -> a Lucide icon slug, by whole-word keyword (simple stemming
    for plurals/verb forms); "circle-dot" when nothing matches. Matching is
    per-word, never a substring of the raw label -- "food"'s "eat" keyword
    can never fire on "Great support" (which merely contains "eat" inside
    "gr-EAT"), the bug that first exposed the old substring check."""
    words = _WORD_RE.findall((label or "").lower())
    if not words:
        return _DEFAULT_ICON
    for keywords, icon in _ICON_KEYWORDS:
        for word in words:
            if any(_word_matches(word, keyword) for keyword in keywords):
                return icon
    return _DEFAULT_ICON


# --- FlowDiagram -------------------------------------------------------------

def build_flow_diagram(beat: Beat, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """``flow_diagram`` beat -> a FlowDiagram job."""
    data = beat.data or {}
    steps = [s for s in (data.get("steps") or []) if isinstance(s, str) and s.strip()][:6]
    if len(steps) < 2:
        return None
    layout = data.get("layout", "chain")
    if layout not in _LAYOUTS:
        layout = "chain"
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    return {
        "composition": "FlowDiagram",
        "props": {"theme": ctx.theme, "steps": steps, "layout": layout},
        "width": ctx.width, "height": ctx.height, "fps": ctx.fps, "frames": frames,
        "transparent": False,
    }


def render_flow_fallback(beat: Beat, width: int, height: int):
    """A static stand-in for a failed FlowDiagram render: the same steps as a
    vertical numbered list (graphics.render_numbered_point) -- the closest
    existing PIL card to "boxes in order" the Architecture names as a
    fallback. Never raises -- motion.render_fallback treats any exception as
    'no fallback for this beat'."""
    from PIL import Image

    from . import graphics as graphics_stage

    data = beat.data or {}
    steps = [s for s in (data.get("steps") or []) if isinstance(s, str) and s.strip()][:6] or ["Step"]
    image = Image.new("RGB", (width, height), graphics_stage.CANVAS)
    rows = [graphics_stage.render_numbered_point(str(i + 1), step, height) for i, step in enumerate(steps)]
    gap = height * 0.03
    total_h = sum(r.height for r in rows) + gap * (len(rows) - 1)
    y = (height - total_h) / 2
    for row in rows:
        x = (width - row.width) / 2
        image.paste(row, (int(x), int(y)), row)
        y += row.height + gap
    return image


# --- IconList -----------------------------------------------------------------

def build_icon_list(beat: Beat, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """``icon_list`` beat -> an IconList job. A `+`/`-` prefix on a label
    (pros_cons mode) decides its side and is stripped before it becomes the
    on-screen label; an unprefixed label in pros_cons mode is a pro."""
    data = beat.data or {}
    raw_items = [s for s in (data.get("items") or []) if isinstance(s, str) and s.strip()][:6]
    if len(raw_items) < 2:
        return None
    mode = data.get("mode", "list")
    if mode not in _MODES:
        mode = "list"
    items: List[Dict[str, Any]] = []
    for raw in raw_items:
        text = raw.strip()
        side: Optional[str] = None
        if mode == "pros_cons":
            if text.startswith("+"):
                side, text = "pro", text[1:].strip()
            elif text.startswith("-"):
                side, text = "con", text[1:].strip()
            else:
                side = "pro"
        items.append({"label": text, "icon": icon_for_label(text), "side": side})
    duration_s = _beat_duration(beat)
    frames = max(1, round(duration_s * ctx.fps))
    return {
        "composition": "IconList",
        "props": {"theme": ctx.theme, "items": items, "mode": mode},
        "width": ctx.width, "height": ctx.height, "fps": ctx.fps, "frames": frames,
        "transparent": False,
    }


def render_icon_list_fallback(beat: Beat, width: int, height: int):
    """A static stand-in for a failed IconList render: the same labels as
    pills (graphics.render_pill), a pro pill in the accent colour in
    pros_cons mode since the PIL card has no tick/cross badge of its own.
    Never raises -- motion.render_fallback treats any exception as 'no
    fallback for this beat'."""
    from PIL import Image

    from . import graphics as graphics_stage

    data = beat.data or {}
    mode = data.get("mode", "list")
    raw_items = [s for s in (data.get("items") or []) if isinstance(s, str) and s.strip()][:6] or ["Item"]
    image = Image.new("RGB", (width, height), graphics_stage.CANVAS)
    pills = []
    for raw in raw_items:
        text = raw.strip()
        accent = False
        if mode == "pros_cons" and text.startswith(("+", "-")):
            accent = text.startswith("+")
            text = text[1:].strip()
        pills.append(graphics_stage.render_pill(text, height, accent=accent))
    gap = height * 0.025
    total_h = sum(p.height for p in pills) + gap * (len(pills) - 1)
    y = (height - total_h) / 2
    for pill in pills:
        x = (width - pill.width) / 2
        image.paste(pill, (int(x), int(y)), pill)
        y += pill.height + gap
    return image
