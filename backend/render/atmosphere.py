"""Atmosphere effects — rain, snow, lightning, sunlight, light leaks, fog, grain.

Everything here is generated procedurally from lavfi sources; no stock footage or
overlay plates are needed, so an effect costs nothing but filter time and works at
any resolution.

Three things were learned the hard way while building these, and all three are
load-bearing:

1. **Blend in RGB, never in YUV.** `screen` is defined on light intensities. Applied
   to YUV chroma planes — which are signed offsets around 128 — screening two
   neutral greys gives 191, and the whole picture turns magenta. Every composite
   here converts to `gbrp` first.
2. **`gradients` needs 8-digit hex.** `c0=0xFF7A2A` is parsed with alpha 0 and
   silently yields a transparent (black) gradient; `0xFF7A2AFF` is what was meant.
   Named colours work too.
3. **Noise sits around its base value.** On a black source `noise=alls=60` never
   exceeds ~148, so a threshold above that produces an empty layer. Rain and snow
   start from grey and threshold near 230, which leaves the sparse bright specks
   that become drops.
"""

import logging
from typing import Dict, List, Optional, Union

from timeline.schema import AtmosphereEffect

logger = logging.getLogger("atmosphere")

# Effects that only add light. Their colour is irrelevant, so they are built as
# grey layers — cheaper, and it keeps them from tinting the picture.
_LUMA_EFFECTS = {"rain", "snow", "lightning", "fog", "grain"}


def _f(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def available() -> List[Dict[str, object]]:
    """Catalogue for the UI."""
    return [
        {"id": "rain", "label": "Rain", "description": "Falling streaks, screened over the picture"},
        {"id": "snow", "label": "Snow", "description": "Slow drifting flakes"},
        {"id": "lightning", "label": "Lightning", "description": "Periodic storm flashes"},
        {"id": "sunlight", "label": "Sunlight", "description": "Warm bloom from a corner"},
        {"id": "light_leak", "label": "Light Leak", "description": "Analogue colour wash across the frame"},
        {"id": "fog", "label": "Fog", "description": "Soft drifting haze"},
        {"id": "wind", "label": "Wind", "description": "Drifting haze with a slow sway"},
        {"id": "grain", "label": "Film Grain", "description": "Fine moving grain"},
        {"id": "flash", "label": "Flash Frame", "description": "A white (or coloured) hit; window it to a few frames"},
        {"id": "shake", "label": "Camera Shake", "description": "The frame jolts randomly; window it to a hit"},
        {"id": "glitch", "label": "Glitch", "description": "Colour planes tear apart on random frames"},
        {"id": "vhs", "label": "VHS / Found Footage", "description": "Scanlines, chroma smear and grain"},
        {"id": "flicker", "label": "Light Flicker", "description": "A failing light: brightness wobbles"},
    ]


def _rain_layer(width: int, height: int, fps: float, duration: float,
                intensity: float, speed: float) -> str:
    # A short noise field stretched tall turns each speck into a streak; scrolling
    # the tall field past the frame makes them fall.
    density = int(round(238 - 12 * intensity))          # lower threshold = more drops
    fall = int(round(500 * speed))
    tall = height * 4
    return (
        f"color=gray:s={width}x{max(60, height // 3)}:r={_f(fps)}:d={_f(duration)},"
        f"noise=alls=60,format=gray,"
        f"lutyuv=y='if(gt(val,{density}),255,0)',"
        f"scale={width}:{tall}:flags=bilinear,"
        f"lutyuv=y='min(255,val*2)',"
        f"crop={width}:{height}:0:'mod(t*{fall},{tall - height})'"
    )


def _snow_layer(width: int, height: int, fps: float, duration: float,
                intensity: float, speed: float) -> str:
    density = int(round(244 - 10 * intensity))
    fall = int(round(110 * speed))
    tall = height * 4
    return (
        f"color=gray:s={max(80, width // 4)}x{height}:r={_f(fps)}:d={_f(duration)},"
        f"noise=alls=60,format=gray,"
        f"lutyuv=y='if(gt(val,{density}),255,0)',"
        f"scale={width}:{tall}:flags=bicubic,gblur=sigma=1.6,"
        f"lutyuv=y='min(255,val*6)',"
        f"crop={width}:{height}:0:'mod(t*{fall},{tall - height})'"
    )


def _lightning_layer(width: int, height: int, fps: float, duration: float,
                     intensity: float, speed: float) -> str:
    # Built tiny and scaled up: the flash is uniform, so per-pixel geq work at full
    # resolution would be wasted. Two spikes per cycle reads as a real strike.
    period = _clamp(3.4 / max(0.15, speed), 0.8, 12.0)
    strength = _clamp(0.55 + 0.45 * intensity, 0.2, 1.0)
    flash = (f"max(0,1-14*abs(mod(T,{_f(period)})-{_f(period * 0.27)}))*{_f(strength)}"
             f"+max(0,1-18*abs(mod(T,{_f(period)})-{_f(period * 0.35)}))*{_f(strength * 0.7)}")
    return (
        f"color=gray:s=32x18:r={_f(fps)}:d={_f(duration)},format=gray,"
        f"geq=lum='255*({flash})',"
        f"scale={width}:{height}:flags=neighbor"
    )


def _sunlight_layer(width: int, height: int, fps: float, duration: float,
                    intensity: float, speed: float, color: str) -> str:
    return (
        f"gradients=s={width}x{height}:r={_f(fps)}:d={_f(duration)}:"
        f"c0={color}:c1=0x000000FF:type=radial:speed={_f(0.004 * speed)}:"
        f"x0={int(width * 0.22)}:y0={int(height * 0.2)}"
    )


def _light_leak_layer(width: int, height: int, fps: float, duration: float,
                      intensity: float, speed: float, color: str) -> str:
    return (
        f"gradients=s={width}x{height}:r={_f(fps)}:d={_f(duration)}:"
        f"c0={color}:c1=0x000000FF:type=linear:speed={_f(0.015 * speed)},"
        f"gblur=sigma=24"
    )


def _fog_layer(width: int, height: int, fps: float, duration: float,
               intensity: float, speed: float) -> str:
    return (
        f"gradients=s={width}x{height}:r={_f(fps)}:d={_f(duration)}:"
        f"c0=0xDFE8F0FF:c1=0x101418FF:type=linear:speed={_f(0.005 * speed)},"
        f"gblur=sigma=30"
    )


def _wind_layer(width: int, height: int, fps: float, duration: float,
                intensity: float, speed: float) -> str:
    return (
        f"gradients=s={width}x{height}:r={_f(fps)}:d={_f(duration)}:"
        f"c0=0xD8E2EAFF:c1=0x14181CFF:type=linear:speed={_f(0.06 * speed)},"
        f"gblur=sigma=26"
    )


def build_effect(
    effect: AtmosphereEffect,
    input_label: str,
    output_label: str,
    width: int,
    height: int,
    fps: float,
    duration: float,
    index: Union[int, str],
) -> List[str]:
    """Filter statements compositing one effect onto `input_label`.

    Returns complete `[in]…[out]` statements ready to join with ";".

    `index` only has to be unique within the graph — it names this effect's
    intermediate labels. Adjustment layers pass a compound id like "2_0" so their
    effects cannot collide with the programme-wide ones.
    """
    kind = (effect.type or "").lower()
    intensity = _clamp(effect.intensity, 0.0, 1.0)
    speed = _clamp(effect.speed, 0.1, 4.0)
    duration = max(0.5, duration)

    # Grain needs no layer — it is applied straight to the picture.
    if kind == "grain":
        amount = int(round(4 + 26 * intensity))
        return [f"{input_label}noise=alls={amount}:allf=t+u{output_label}"]

    # --- the treatments: no layer, the picture itself is disturbed -----------
    # These are what a horror edit reaches for at a hit. They are meant to be
    # windowed by an adjustment clip (a flash is two frames, a shake half a
    # second); over a whole programme they would be unwatchable.
    if kind == "flash":
        colour = (effect.color or "white").replace("0x", "#")[:7] if effect.color else "white"
        alpha = _f(_clamp(0.5 + 0.5 * intensity, 0.0, 1.0))
        return [f"{input_label}drawbox=color={colour}@{alpha}:t=fill{output_label}"]

    if kind == "shake":
        # Crop a window that wanders randomly every frame, scale back up. The
        # amplitude scales with intensity; `random` is re-seeded per axis so the
        # two do not move together.
        amp = max(2, int(round(width * (0.006 + 0.03 * intensity))))
        return [
            f"{input_label}crop=w=iw-{2 * amp}:h=ih-{2 * amp}:"
            f"x='{amp}+{amp}*(random(1)-0.5)*2':y='{amp}+{amp}*(random(2)-0.5)*2',"
            f"scale={width}:{height}:flags=bilinear,setsar=1{output_label}"
        ]

    if kind == "glitch":
        # Red and blue planes torn apart on a random fifth of the frames, with
        # a burst of noise on the same frames so the tear reads as damage.
        shift = max(2, int(round(width * (0.003 + 0.012 * intensity))))
        chance = _f(0.08 + 0.3 * intensity)
        gate = f"lt(random(3),{chance})"
        return [
            f"{input_label}rgbashift=rh=-{shift}:bh={shift}:enable='{gate}',"
            f"noise=alls={int(round(30 + 40 * intensity))}:allf=t:enable='{gate}'{output_label}"
        ]

    if kind == "vhs":
        # Scanlines, softened chroma pushed sideways, grain and a slight
        # desaturation: found-footage in one chain. geq costs real time at
        # full resolution, so the scanline pass runs at 1/3 height first.
        dark = _f(1.0 - (0.18 + 0.2 * intensity))
        return [
            f"{input_label}chromashift=cbh=-{2 + int(3 * intensity)}:crh={2 + int(3 * intensity)},"
            f"eq=saturation={_f(0.85 - 0.2 * intensity)}:contrast={_f(1.0 + 0.08 * intensity)},"
            f"noise=alls={int(round(10 + 24 * intensity))}:allf=t+u,"
            f"geq=lum='if(mod(Y\\,3)\\,lum(X\\,Y)\\,lum(X\\,Y)*{dark})':cb='cb(X\\,Y)':cr='cr(X\\,Y)'"
            f"{output_label}"
        ]

    if kind == "flicker":
        # A failing light: brightness wobbles on two incommensurate sines.
        depth = _f(0.03 + 0.09 * intensity)
        rate = _f(23.0 * speed)
        return [
            f"{input_label}eq=brightness='-{depth}*0.5+{depth}*sin(t*{rate})*sin(t*{_f(7.3 * speed)})':"
            f"eval=frame{output_label}"
        ]

    # Wind sways the frame as well as hazing it, which is what sells the movement.
    if kind == "wind":
        drift_x = 0.012 + 0.03 * intensity
        drift_y = drift_x * 0.45
        over_w = int(round(width * (1 + drift_x * 2)))
        over_h = int(round(height * (1 + drift_y * 2)))
        over_w += over_w % 2
        over_h += over_h % 2
        sway = (
            f"{input_label}scale={over_w}:{over_h},"
            f"crop={width}:{height}:"
            f"'{(over_w - width) // 2}+{int((over_w - width) * 0.45)}*sin(t*1.7)':"
            f"'{(over_h - height) // 2}+{int((over_h - height) * 0.45)}*sin(t*1.1)'"
            f"[wind_sway{index}]"
        )
        layer = _wind_layer(width, height, fps, duration, intensity, speed)
        return [
            sway,
            f"{layer},format=gbrp[wind_haze{index}]",
            f"[wind_sway{index}]format=gbrp[wind_base{index}]",
            # shortest=1: the generated layer is built long enough to cover the
            # programme, and without this blend would stretch the *programme* out
            # to match the layer instead of the other way round.
            f"[wind_base{index}][wind_haze{index}]"
            f"blend=all_mode=screen:shortest=1:all_opacity={_f(0.10 + 0.25 * intensity)},"
            f"format=yuv420p{output_label}",
        ]

    builders = {
        "rain": lambda: _rain_layer(width, height, fps, duration, intensity, speed),
        "snow": lambda: _snow_layer(width, height, fps, duration, intensity, speed),
        "lightning": lambda: _lightning_layer(width, height, fps, duration, intensity, speed),
        "fog": lambda: _fog_layer(width, height, fps, duration, intensity, speed),
        "sunlight": lambda: _sunlight_layer(width, height, fps, duration, intensity, speed,
                                            effect.color or "0xFFE9B0FF"),
        "light_leak": lambda: _light_leak_layer(width, height, fps, duration, intensity, speed,
                                                effect.color or "0xFF7A2AFF"),
    }
    build = builders.get(kind)
    if build is None:
        logger.warning("Unknown atmosphere effect %r; skipping", effect.type)
        return []

    opacity = {
        "rain": 0.25 + 0.5 * intensity,
        "snow": 0.35 + 0.55 * intensity,
        "lightning": 0.45 + 0.5 * intensity,
        "fog": 0.12 + 0.35 * intensity,
        "sunlight": 0.15 + 0.45 * intensity,
        "light_leak": 0.12 + 0.4 * intensity,
    }[kind]

    layer = f"layer{index}"
    # gbrp on both sides: screen is an RGB operation, and running it over YUV
    # chroma is what turns the whole frame magenta.
    return [
        f"{builders[kind]()},format=gbrp[{layer}]",
        f"{input_label}format=gbrp[atmo_base{index}]",
        # shortest=1 is load-bearing: the layer is deliberately generated a little
        # longer than the programme, and blend otherwise pads the programme out to
        # the layer's length — which silently undid every transition's shortening
        # and left the video running 1.2s past its own audio.
        f"[atmo_base{index}][{layer}]"
        f"blend=all_mode=screen:shortest=1:all_opacity={_f(_clamp(opacity, 0.0, 1.0))},"
        f"format=yuv420p{output_label}",
    ]


def build_aspect_bars(ratio: float, width: int, height: int,
                      input_label: str, output_label: str) -> Optional[str]:
    """Letterbox to a cinematic aspect ratio, keeping the output resolution.

    Crops the picture to the target shape and pads the black bars back on, so the
    file stays 16:9 (or whatever it was) and simply *looks* like 2.39:1.
    """
    if ratio <= 0:
        return None
    current = width / max(1, height)
    if abs(current - ratio) < 0.01:
        return None

    if ratio > current:                      # wider target: bars top and bottom
        keep = int(round(width / ratio))
        keep -= keep % 2
        keep = max(2, min(height, keep))
        if keep >= height:
            return None
        return (f"{input_label}crop={width}:{keep}:0:{(height - keep) // 2},"
                f"pad={width}:{height}:0:{(height - keep) // 2}:color=black{output_label}")

    keep = int(round(height * ratio))         # taller target: bars left and right
    keep -= keep % 2
    keep = max(2, min(width, keep))
    if keep >= width:
        return None
    return (f"{input_label}crop={keep}:{height}:{(width - keep) // 2}:0,"
            f"pad={width}:{height}:{(width - keep) // 2}:0:color=black{output_label}")
