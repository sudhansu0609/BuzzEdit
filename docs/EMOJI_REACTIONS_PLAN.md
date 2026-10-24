# Emoji reactions (presentation pass)

Design:
- `backend/presentation/emoji_fx.py` matches spoken words (Devanagari, romanised Hindi, English; NFC, nukta-folded, punctuation-stripped) against one table, `EMOJI_VOCAB`, plus a percent rule (100% -> 💯, other % -> 📊).
- `plan_reactions` ranks matches by `emphasis_z` (+ bonus for percent/big-number entities, + the category's own weight); generic words ("no", "kya", "time", "kam") need `emphasis_z >= WEAK_MIN_EMPHASIS`. Ties broken by a project-seeded RNG.
- Rules: >= MIN_GAP_S (3 s) between one reaction's end and the next start; at most 6 x `emoji_density` per rolling minute; never in a busy window (cards, newspaper/document cutaways, text_fx, overlay graphics, popups); during other B-roll only in the top-right corner at a smaller size.
- Placement: beside the head on the side with more room (away from the face centre), 0.18 of the canvas height, pushed up out of the caption band (caption preset `pos_y` +- CAPTION_HALF_BAND).
- Render: each emoji drawn once with Pillow from Segoe UI Emoji (`embedded_color=True`) into a cached RGBA PNG with a soft baked shadow (`assets/graphics/emoji_<cp>_<size>.png`).
- Animation: the timeline's scale animation (`scale_end`) promotes an overlay to a full black canvas (render/effects.py), so the pop is baked as stepped items sharing one source: 4 scale steps 0.3 -> 0.7 -> 1.12 -> 1.04 over 0.25 s, a 1.2 s hold at 1.0, then 2 opacity steps over 0.2 s. All items carry origin `emoji_fx` on one fresh V track above B-roll (text/caption tracks always draw over V tracks).
- Settings: `emoji_reactions: "auto"|"on"|"off"` (auto = on only for infotainment-type genres: general, comedy, gaming, tech, science_education, finance, motivational, health_fitness, cooking, travel, vlog), `emoji_density`, `emoji_style` ("pop"). Report: `emoji_reactions`.

## Status

| # | Deliverable | Status |
|---|---|---|
| 1 | emoji_fx.py (vocab, render_emoji, plan_reactions, apply_reactions) | done: matching (Devanagari/romanised/English, phrases, %, ₹) smoke-tested; contact sheet renders in colour with shadow |
| 2 | Director wiring + settings + report field | done: settings emoji_reactions(str auto/on/off)/emoji_density/emoji_style; report.emoji_reactions; stage after text_fx via emoji_fx.plan_and_apply (clears first) |
| 3 | tests/test_emoji_fx.py + existing presentation tests | done (cut to 3 tests per user): vocabulary, placement rules, director add/clear; with test_text_fx/test_presentation/test_graphics/test_verify_presentation all pass |
| 4 | Visual check on 5fbc856f (15 s window, 3 planted reactions) | done: window 88-103 s (252-267 s fails in ffmpeg even with no emoji, pre-existing). Emoji in full colour, crisp, right of the head at eye level, overshoot visible mid-pop, well clear of the caption band. Frames: scratchpad/emoji/frame_pop.jpg, frame_hold.jpg, frame_hold3.jpg |
| 5 | This plan doc | done |

## Frontend Status

| # | Added | Request |
|---|---|---|
| 1 | AgentPanel Dressing section: "Emoji reactions" select (auto/on/off, default auto) and "Emoji density" slider 0.5-2.0 step 0.1 showing max reactions/min (6 x density); disabled when off | `POST /api/presentation/{project}/start` `{settings: {..., emoji_reactions, emoji_density}}` |
