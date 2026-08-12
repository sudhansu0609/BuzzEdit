TEMPLATES = {
    "vlog": {
        "name": "YouTube Vlog",
        "transition_type": "xfade",
        "transition_duration": 0.5,
        "color_preset": "vibrant",
        "aspect_ratio": "16:9",
        "burn_captions": True,
    },
    "tutorial": {
        "name": "Tutorial & Demo",
        "transition_type": "zoomin",
        "transition_duration": 0.4,
        "color_preset": "cinematic",
        "aspect_ratio": "16:9",
        "burn_captions": True,
    },
    "shorts": {
        "name": "Shorts / Reels (9:16)",
        "transition_type": "xfade",
        "transition_duration": 0.3,
        "color_preset": "vibrant",
        "aspect_ratio": "9:16",
        "burn_captions": True,
    },
    "podcast": {
        "name": "Podcast & Interview",
        "transition_type": "dissolve",
        "transition_duration": 0.75,
        "color_preset": "warm",
        "aspect_ratio": "16:9",
        "burn_captions": True,
    },
}


def get_template_settings(template_id: str) -> dict:
    return TEMPLATES.get(template_id.lower(), TEMPLATES["vlog"])
