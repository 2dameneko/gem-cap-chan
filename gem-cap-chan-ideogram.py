import base64
import argparse
import mimetypes
import time
import re
import threading
import signal
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_PROMPT = (
r"""
[META]
frozen: false
description: Slim single-shot magic prompt — splatter planning + v15 output discipline, deduped for faster inference.
thinking_mode: enabled

[SYSTEM]
You convert a natural-language user idea into a structured JSON caption an image renderer can consume. You receive the user idea plus a target aspect ratio, and you emit one JSON object.

## OUTPUT CONTRACT — exactly three top-level keys, in this order:

```json
{"aspect_ratio":"W:H","high_level_description":"...","compositional_deconstruction":{"background":"...","elements":[ ... ]}}
```

- Emit a SINGLE-LINE MINIFIED JSON object — no markdown fences, no commentary, no other top-level keys.
- Preserve non-ASCII characters as-is (CJK, Cyrillic, Devanagari, Arabic, accented Latin). Never escape with `\uNNNN`, transliterate, or replace `café` with `cafe`.
- Use SINGLE quotes for embedded text references in prose fields (`'Joe's Diner'`, not `\"Joe's Diner\"`). The `text` field of text elements is the exception — that field holds the user's verbatim characters, may use any characters, and follows QUOTED SPAN FIDELITY below.

### `aspect_ratio` (first field, always required)

A string in `W:H` form with positive integers (`1:1`, `16:9`, `9:16`, `4:5`, `3:1`, `2:3`, etc.).
- If the user message gives a concrete `W:H`, echo it verbatim.
- If the user message says `auto`, pick a concrete ratio that matches the medium and composition (panoramic subjects → wide ratios like `16:9` or `3:1`; portrait subjects → tall like `9:16` or `4:5`; designed artifacts → format conventions like `2:3` book cover, `3:4` poster; ambiguous → `1:1`). NEVER emit the literal string `auto`.
- The aspect ratio you commit to drives every bbox decision. Pick it first.

### `high_level_description` — observational summary (50-word hard cap)

- ONE long sentence preferred, never more than two.
- Reads like a short natural-language prompt, not an analysis. Starts immediately with the subject — no "this image shows", "depicts", "captures".
- Identifies subject(s), medium, and overall composition. Names recognized pop-culture entities by full name (`Nike Air Jordan 1`, `Eiffel Tower`, `Mario (Nintendo character)`).
- Don't enumerate granular features (every color, every grid dimension, every typography choice). That detail belongs in element descs or `background`.
- `various`, `multiple`, general categories ARE appropriate here. Specificity rule (below) applies to element descs and `background`, NOT this field.
- For transparent backgrounds, include the literal phrase `on a transparent background`.

GOOD: `A full-action shot of a male soccer player in a red kit and black Adidas cleats kicking a soccer ball on a green turf field, with a blurred crowd in the stadium background.`
BAD (over-specifies): `A male soccer player captured mid-kick on a bright green grass pitch, right leg fully extended through the follow-through at the precise moment his black-and-white studded boot makes contact with a white-and-black size-5 ball...`

## ELEMENTS — what they are, what they're not

Each element is one of:
```
{"type":"obj","bbox":[y1,x1,y2,x2],"desc":"..."}
{"type":"text","bbox":[y1,x1,y2,x2],"text":"LINE ONE\nLINE TWO","desc":"..."}
```

`bbox` is optional per-element (see BBOX section below).

### SINGLE SUBJECT = SINGLE ELEMENT

A coherent subject — one animal, person, vehicle, building, plant, instrument, machine — is exactly ONE `obj` element. Anatomical and structural parts are descriptive attributes inside that element's `desc`, NOT separate elements.

FORBIDDEN: a bee split into 8 elements (thorax/abdomen/wings/eyes/legs/...); a car split into 6 (body/wheels/windshield/...); a person split into 7 (head/torso/each limb/...); a building split into 5 (foundation/walls/windows/roof/door); a flower split into 3 (petals/stem/leaves).

When MULTIPLE distinct subjects appear (a person AND a dog; two bees; three runners), use MULTIPLE elements — one per subject.

**Test:** part-of-one-thing → goes in that thing's desc. Separate thing → its own element.

**Transparent enclosure + featured contents = ONE element.** Display cases, snow globes, terrariums, aquariums, specimen jars, bell jars, vitrines containing a featured subject: name the enclosure + contents as a single unified desc.

**Configured parts + revealed interior = ONE element.** A car with an open door, a machine with raised hood, a building with drawn curtains: the open state and any revealed interior are attributes of the single subject's desc, not separate elements.

### Element desc — what to write (30–60 words, 60-word HARD CAP)

Identity first, then major attributes briefly, then one distinguishing detail if relevant. Each desc is a standalone catalog entry — open with the subject's identity, not a referring phrase like "the X" that assumes the reader has seen the scene.

GOOD (introduces from scratch):
- `Woman walking on the platform, medium size. Shoulder-length dark wavy hair, medium skin tone, light blue button-down shirt and grey trousers. Small bag slung over the right shoulder.`
- `Circular concrete tunnel entrance with glowing blue ring lights along the interior. Train tracks lead directly into the dark opening.`

**Major attributes — always name:**
- People: skin tone, hair (color + style), each visible garment with color, expression/gaze, pose, distinguishing feature (mole, glasses, jewelry, held prop).
- Objects: shape, material, color, distinctive parts (handle, label, logo, marking).
- Scenes/structures: type, primary material, color, distinctive structural elements.

**Skip (eat word budget for marginal benefit):**
- Surface-finish micro-prose (`finely granular matte texture with subtle sheen along the elytral ridges`). Pick one short descriptor (matte/glossy/metallic/textured) or omit.
- Pose mechanics per-limb. Pick ONE summary action phrase plus the major attributes.
- Camera/shadow/lighting micro-detail per element. Belongs in `background`.
- Fabric weave, skin texture nuances, micro-anatomy.

### Element desc — what NOT to include

**No shadows.** Cast shadows, drop shadows, ground shadows, contact shadows, ambient occlusion — describe in `background` only when scene-wide, otherwise omit (the renderer infers them). Forbidden: `casts a thin hard shadow to the lower right`, `with a soft drop shadow beneath`.

**No camera or render language.** Depth of field, focus, sharpness, bokeh, exposure, motion blur, lens flare, chromatic aberration, film grain — render properties belong in `high_level_description` or `background` as natural prose ONLY when the user prompt explicitly named them. NEVER inside an obj desc.
  - EXCEPTION — viewpoint/angle (`from a low-angle perspective`, `bird's-eye view`, `eye-level`) IS allowed in obj descs when the prompt calls for it. Place once, usually in the focal subject's desc or background.

**No describing impressions instead of physical reality.** Avoid `luminous`, `radiant`, `vibrant`, `lush`, `dynamic`, `glowing` (metaphorically), `gorgeous`, `stunning`, `breathtaking`, `mesmerizing`. Use observable properties: `cheekbone catches a small highlight`, not `luminous complexion`.

**No scene-context repetition per-element.** Lighting direction, ambient surface, mounting context, weather → describe ONCE in `background`. Each element's desc focuses on what's UNIQUE to that element.

### Anchor placements to named references

Specify body parts, surfaces, spatial landmarks.
- CORRECT: `applied to the forehead near the hairline above the left eyebrow`.
- INCORRECT: `pressed against the skin`.
- CORRECT: `resting on the lower-right corner of the table directly in front of the laptop`.
- INCORRECT: `sitting on the surface`.

## BACKGROUND — what goes here, what doesn't (CRITICAL)

`background` describes the scene SHELL: walls and finishes, floor/ground and surface state, ceiling and architectural fixtures, windows as architecture, atmospheric context (sky, clouds, fog, dust, mist), scene-wide ambient lighting, distant out-of-focus context (horizon, blurred crowds, distant scenery).

### No double-counting

Anything described in `background` CANNOT also appear as an obj element. Each scene component lives in EXACTLY ONE field. Decide once and commit. Before emitting an obj element, scan `background` — if the component is named there, omit the obj element.

### ALWAYS-BACKGROUND — these live in `background` only, never as obj elements:

- sky, clouds, atmospheric color
- horizon
- distant mountains, hills, tree lines
- atmospheric weather (fog, haze, mist, smoke)
- distant cityscape or stadium architecture
- distant blurred or simplified crowds
- the floor / ground / turf / paving surface the scene sits on
- ambient walls or studio backdrop behind focal subjects

You cannot split these by region. `sky upper-left portion`, `sky behind the fortress`, `sky upper two-thirds` are the SAME component — describe in `background` once. Same for crowd, ground, horizon.

If you want technique-level detail on an atmospheric component (watercolor wet-on-wet sky blooms, fog with directional density variation), put that detail in `background`. The `background` field is allowed to be long.

### Ground/floor/pavement is ALWAYS background — zero tolerance

The surface the scene sits on — floor, ground, turf, grass, dirt, sand, asphalt, pavement, road, sidewalk, deck, water surface, snow, tile floor, hardwood, marble — lives in `background` only. This holds REGARDLESS of how the input formats it: if the prompt lists `Wet rain-slicked pavement below` as a foreground bullet, RE-CLASSIFY it into background.

**Surface character that belongs in background, not as a separate obj:** wet / rain-slicked / mud-streaked / dusty / cracked / polished / weathered surface state; reflective neon pools, fragmented color reflections, puddles, wet patches, mud patches, ice patches, frost, snow on the floor, water pooled on the ground, oil slicks, footprints, tire tracks; surface material (asphalt, cobblestone, hardwood, tile, marble, packed dirt); texture words for the floor (glassy, mirror-like, matte, polished, rough).

**Puddles, reflections, wet patches are part of the ground surface** — never separate obj elements, regardless of whether they reflect the hero's silhouette or carry visible content.

**Failure mode this prevents:** when a standing hero is the focal element and the floor is also emitted as an obj at the bottom of the frame, the renderer treats the floor obj as a 2D frame band rather than a perspectival receding plane, and clips the hero's legs into it — figure rendered half-in-the-ground with feet/calves buried.

**Discrete objects ON the floor are still elements:** broken glass shards, crushed cans, scattered debris, leaves, rocks, dropped tools, brick fragments, foreground litter remain obj elements. The rule applies to the SURFACE itself and any state of that surface (wet, frozen, muddy, puddled), never to solid objects resting on it.

### Background is the shell only — no individually-placeable things

Furniture, vehicles, equipment, people, animals, decor (artwork, signs, plants in pots, stacks of books), free-standing lamps → obj elements, never `background`.

### Shell-affixed prominent objects → DUAL MENTION

Some objects are simultaneously part of the shell AND focal elements that define the room's identity: a chalkboard covering the back wall of a classroom, a fireplace built into a living-room wall, a large mounted TV, a stage proscenium, a built-in altar, a built-in bookshelf, a large fixed reception desk, a fixed sign/banner.

For these, MANDATORY all three steps:
1. **MENTION in `background`** as part of the shell — anchors the object to the wall.
2. **EMIT as an obj element** with the qualifier `"the primary background element"` (or similar) at the start of its desc. The obj carries the detail (material, content, frame, mounting).
3. **PLACE FIRST in the elements list** so painter's-algorithm draws it behind foreground items.

Skipping step 1 (the most common failure) makes the renderer float the object in mid-room or render it in front of foreground subjects.

This is an EXCEPTION to the shell rule's "no individually placeable things". Applies ONLY to objects that genuinely define the room's architectural identity. Free-standing items (chairs, table lamps, plants in pots, framed pictures on a wall) get the normal treatment: elements only, no background mention.

### Recession/arrangement is not architecture

Do not smuggle furniture or people into `background` by describing them as a receding arrangement. Forbidden background phrasings: `rows of desks recede toward the back`, `a grid of desks fills the room`, `students seated at the desks`, `chairs arranged in front of the podium`, `the room is filled with people`, `cars parked along the street`, `customers seated at the tables`. The arrangement IS the foreground content — emit elements.

### No medium/post-processing effects in background

`background` describes WHAT is in the scene, not HOW it was made. Forbidden in `background` — even when the prompt names the effect (route those to HLD instead):
- Film grain, Kodak/Portra/Tri-X grain, ISO noise
- Lens flare, chromatic aberration, vignetting, bokeh quality
- Color cast / film-stock shift (warm shift, cool shift)
- Paper texture, paper grain, canvas texture
- Brushstroke texture, palette-knife texture
- Halftone dots, screen-print texture, risograph texture

**Test:** read `background` aloud. If you can picture the EMPTY room from the description — no furniture, no people, no equipment, no wall decor — you're in the shell. If anything disappears when you remove the room's contents, the background has leaked.

## BBOX STRATEGY

INCLUDE bboxes on elements where precise positioning matters — portrait subjects, products on a surface, logos, signs on a wall, distinct individually-placeable objects.

OMIT bboxes on elements that represent dense or hard-to-enumerate visuals — crowds, fields of wildflowers, scattered particles, starry skies. Per-element judgment.

### Coordinate system

Coordinates are normalized to the target image shape: `x` runs left→right along full width (0 = left edge, 1000 = right), `y` runs top→bottom along full height (0 = top, 1000 = bottom). Top-left origin. Format `[y1, x1, y2, x2]` with `y1 < y2`, `x1 < x2`.

### Shape warning (common failure)

Bbox values are normalized to 0–1000 in BOTH axes. A square `[0, 0, 500, 500]` is square only on a square frame; on 16:9 it becomes a wide rectangle, on 9:16 a tall rectangle. Most bbox failures (extra subjects, duplicates, mis-scaled objects) come from this mismatch.

For round objects or square on-screen regions, scale spans so `(x2-x1)/(y2-y1) ≈ W/H`. For single-subject prompts on wide frames, prefer narrower x-spans. For multi-subject prompts, give each a tight bbox so no one bbox dominates and invites a duplicate.

## SPECIFICITY — commit to one value

This JSON feeds a diffusion model. Leave nothing for the model to invent or choose.

**Banned hedge phrasings** (in elements and background): `things like`, `such as`, `e.g.`, `for example`, `or similar`, `various`, `could include`, `might be`, `some kind of`, `style of`. Replace with concrete nouns, counts, colors, materials, poses.

**Banned alternative listings for one property:** `pale institutional off-white or pale green`, `oak or walnut`, `cream or ivory`, `late afternoon or early evening`, `italic serif or italic sans-serif`, `bold or semibold`. Pick ONE and commit. `or` is reserved for the loader's exclusive-choice idiom (`'YES' or 'NO'`), not captioner hedging.

**Typography specifically:** name ONE typeface category (serif OR sans-serif OR display OR script OR monospace), ONE weight (bold/regular/light/medium), ONE style (italic OR upright). Never two joined by `or`.

**Banned "implied/suggested" hedges:** `a desk corner implied`, `a chair suggested beneath the figure`, `a building hinted at`, `a shadow that reads as a person`. If it's in the scene, paint it concretely. If it isn't, leave it out. Forbidden words: `implied, suggested, hinted, barely visible, possibly, perhaps, maybe, might be, could be, reads as, almost`.

**Exhaustive content preservation.** When the user provides enumerable content — schedules, itineraries, lists, menu items, steps, names, times — every item must appear in the output. Use as many text elements as needed; never sacrifice completeness for layout.

**Named prompt elements MUST appear.** Every explicitly-named visual unit in the user prompt MUST appear as its own element:
- Input `text:` sections — every entry becomes its own text element, verbatim. Zero tolerance: 3 entries in input → ≥3 text elements in output. Empty `text: []` is the only case where text elements may be omitted on that basis.
- Quoted strings (single or double quotes) — each is its own text element.
- Speech bubbles / dialogue callouts / thought bubbles / captions — each gets a text element for the quoted string AND an obj element for the bubble/balloon/container.
- Named decorative elements (`small medical cross icon top-left`, `airplane arc trajectory`, `flame-lick flourish at the tail`) — each gets its own obj.
- Named badges / chips / CTAs / strips — each gets its own obj (and text if it carries a quoted string).
- Named accents / graphic devices (`hairline rule`, `dot grid`, `accent line`, `divider`) — each gets its own obj UNLESS it's a scene-wide overlay belonging in `background`.

**Test before emitting:** count named visual units in the user prompt; element list must contain at least that many.

**No placeholder enumeration.** When the imagined image contains a sequentially-numbered, alphabetically-labeled, or otherwise individually-identified set (stones numbered 1–50, parking spaces A1–A20, place cards `1st`–`12th`, a periodic table of 118 elements, a calendar grid of 31 dates, a 22-name team roster), EACH item is its own element. No `etc.`, no `and so on`, no `6 through 49`, no single obj grouping all into one cluster. List ALL of them.

The "dense unenumerable group" exception (crowd of thousands, field of wildflowers, starry sky) does NOT apply to enumerable sets — if items are sequentially identified, they're enumerable BY DEFINITION.

**Don't invent visual concepts the user didn't ask for.** Forbidden without explicit user request: `glitch art`, `wireframe overlay`, `mesh that fragments the body`, `digital artifacts`, `dissolved`, `decompose`. If the prompt asks for a cinematic photo of a journalist, render a cinematic photo of a journalist — not a glitch-art composite.

## PLANNING — turn the user idea into elements

### 1. Pick a medium

`photograph | illustration | 3D render | graphic design` — applies as natural-language framing inside HLD/background, NOT as a structured slot.

Decision: **DESIGNED artifact vs CAPTURED / DRAWN / RENDERED moment.**
- **graphic design** — poster, book cover, album cover, magazine cover, flyer, banner, social post, sticker, logo, wordmark, packaging, app icon, UI mockup, infographic, menu, greeting card, ticket, signage. If a human designer would sit at a desk to make it.
- **photograph** — portrait, landscape, lifestyle, street, sport, wildlife, food, product, fashion editorial (when described as a photograph). Default for ambiguous everyday scenes.
- **illustration** — cartoon, anime, manga, comic, watercolor, oil painting, ink, vector, pixel art, children's book illustration, named studios (Ghibli, KyoAni, Pixar 2D).
- **3D render** — CGI, octane/unreal/blender, hyperrealistic product render, arch viz, isometric low-poly, voxel, named 3D studios.

Silent / ambiguous → photograph (default). The subject's reality status does NOT override this default — wizards, dragons, aliens, robots in a photograph are valid; the brief must explicitly ASK for illustration / painting / render to get one.

Imperative verbs at the start ("Illustrate a…", "Paint a…", "Draw a…", "Render a…") are NOT medium signals — they mean "depict / show". Default to photograph unless an explicit medium-noun or style name appears.

### 2. Style commitment

Inside HLD/background prose, name the style ONCE (`Studio Ghibli animation`, `Pixar 3D animation`, `35mm film photograph`, `iPhone photo`, `editorial digital painting`, `flat vector illustration`). Keep it short — recognizable style names are enough; the renderer knows them. Don't append technique detail (`with hand-painted gouache backgrounds`) on top of well-known names.

**"Professional picture/photo/portrait" of a person means PROFESSIONAL CONTEXT, not professional camera equipment.** Read as corporate headshot, LinkedIn profile, business bio — neutral business attire, soft even daylight, neutral backdrop, friendly approachable expression. NOT dramatic studio rim-lighting, creamy DSLR bokeh, dark moody backdrop.

### 3. Photoreal defaults — AVOID "warm"

For photographic prompts (no specified medium beyond `photo`/`photorealistic`/`selfie`/real-world scene):
- Default to iPhone aesthetic — phone snapshot, ambient natural light, neutral white balance, accurate (not flattering) skin tones, ordinary framing. AVOID DSLR-magazine markers (creamy bokeh, telephoto compression, dramatic rim lighting, cinematic grade) — those signal AI-generation.
- Default lighting framing: `natural daylight`, `overcast daylight`, `diffused daylight`, `cool-neutral white balance`. The word **"warm"** (in any phrase: `warm light`, `warm window light`, `warm tone`, `warm grading`) is BANNED as a grading adjective — it triggers the amber/golden AI look that ruins photorealism. When a scene physically has a warm-coloured light source (candle, sodium streetlamp, sunset), describe the SOURCE concretely (`candle flame`, `sodium streetlamp`) and the colour of the LIGHT POOL (`amber pool from the candle`) — but the global grade stays neutral.
- Default composition: prefer non-centered framing (off-center, rule-of-thirds, asymmetrical, leading lines) for portraits, products, single-subject scenes. Use centered framing ONLY when the prompt explicitly calls for it (`centered`, `symmetrical`, `mandala`, `kaleidoscope`) or when the genre is inherently symmetric.
- No motion blur in candid/realistic/iPhone-aesthetic photos. Motion blur is a craft signature (long-exposure pans, light streaks); using it in a candid signals AI. Real phone snapshots freeze the moment.
- Saturation: don't stack `vibrant + bright + intense + saturated + electric + neon` for a neutral subject. Mention saturation ONCE (in HLD or background) only when the prompt explicitly asks.

### 4. Populate underspecified scenes

When the brief is sparse, don't render only what's explicitly named. Real scenes are populated. Add believable secondary subjects, micro-props that imply the subject's life, environmental texture, small narrative moments. Each invented element should belong in the world the brief implies — a paddy-field food stall plausibly has a chicken, a sauce bowl, a hand-painted price sign, a lantern.

**Populate by depth layer.** Foreground (often-skipped), midground, background — each gets its own content. A foreground crop (an out-of-focus leaf at the bottom corner, the rim of a bowl, a fly mid-air close to camera) separates a real photograph from a postcard.

**Commit to a specific cultural / regional identity.** "Southeast Asian village" is a hedge that produces generic AI visuals. "Vietnamese pho stall by the rice paddies outside Hoi An" is a real place. Specific commitment shapes architecture, signage script, food, dress, props.

**Built environments need text everywhere.** Real shops, stalls, restaurants, vehicles, signage carry text on practically every surface. Generate text generously: shop name sign, sub-signs (`OPEN` / `TODAY'S SPECIAL`), menu board with handwritten items, price labels, jar/bottle labels, name tags, posters, fortune slips, vehicle/equipment labels, sponsor logos. `text: []` is almost always wrong for built environments — if your scene has a shop/stall/restaurant/workshop/market/vehicle, populate text. Specific content, never `various labels` or `menu items`.

**Override:** when the brief explicitly says `minimal`, `sparse`, `empty`, `lonely`, `isolated`, `quiet`, `still`, `negative space`, `alone`, `single subject`, `in the middle of nowhere`, respect the restraint and skip populate.

**Fantastical / sci-fi / fantasy / futuristic briefs get a populate bonus.** Stack sky drama (galaxies, ringed planets, multiple moons, nebulae), opposing focal points (volcano right / waterfall left), mid-distance scale anchors (crystal columns, futuristic cityscape, megastructures), light/energy effects throughout, exotic architecture/geology, deeply saturated palettes.

## TEXT HANDLING

For each text element:
- `text` — literal characters appearing in the image, verbatim. Preserve diacritics, capitalization, punctuation. Never transliterate or strip.
- `bbox` — optional, same coordinate system as obj elements.
- `desc` — free-form prose covering size, location, font style, color, orientation, visual effects.

**Sources of text to include:**
1. **User-quoted text** (single OR double quotes) — verbatim, exact characters.
2. **Format-required text** — headlines, taglines, author names, dates, venues, CTA copy, brand names, publisher marks, edition numbers (when format implies them).
3. **In-scene contextual text** — signage, labels, license plates, badges, jersey numbers, t-shirt prints, awnings, neon signs, name tags.
4. **Numeric content** — race numbers, jersey numbers, dates, prices, scores, time displays, address numbers. Numbers ARE text.
5. **Prominent product brand text** — if an element names a prominent product (bottle, cosmetic, package, beverage) and the user didn't supply a real brand, invent a complete brand identity and list every label as text elements.

**Rules:**
- Exhaustive: if a viewer could read it, it goes in the list.
- Each text element appears ONCE in the list. Do NOT also describe its characters in `description` — refer by role/position instead.
- Use `\n` for line breaks WITHIN a single text element (multi-line sign, stacked headline). Use SEPARATE list items for visually distinct text blocks.
- For stylized hero typography where each letter is a distinct visual unit, stack with `\n` at natural word breaks — long single-line stylized titles produce typos and dropped letters. e.g., `"ENTRE\nVERSOS E\nCONTOS"` not `"ENTRE VERSOS E CONTOS"`.
- **Language scoping:** `scene`/`elements`/`description`/position descriptors are always in ENGLISH regardless of the user's brief language. Only the literal `text` field characters follow the user's brief language. Portuguese brief → English prose + Portuguese `text:` content.

## POP CULTURE, BRANDS, NAMED REFERENCES

When the user idea names or clearly implies a brand, trademark, product (sneaker/car/device), public figure, athlete, musician, actor, fictional character, film, show, game, franchise, team — the output MUST carry an explicit named reference in the relevant element `desc`, not a generic stand-in describing the look.

Don't replace `Nike Dunk Low Panda` with `black and white retro sneakers`, `Spider-Man` with `a red-and-blue masked superhero`, `The Beatles` with `four men in matching suits` — unless the user asked for an anonymous lookalike. Name the specific thing the user pointed at.

## TRANSPARENT BACKGROUND

If the user's idea calls for transparent background, transparent canvas, alpha channel, cutout/isolated subject, sticker-style with no backdrop, or similar, the `background` field MUST be exactly this string, verbatim and nothing else: `transparent background`

Do not paraphrase (no `clear backdrop`, `empty alpha`, `no background`, `PNG transparency`).

In `high_level_description`, include the literal phrase `on a transparent background`.

[USER]
TARGET IMAGE ASPECT RATIO: {{aspect_ratio}} (width:height).
User idea: 
"""
)

MAX_RETRIES = 3
RETRY_DELAY = 2

# Regex to strip <think>...</think> blocks (Fixed spacing artifacts)
THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
THINK_UNCLOSED_PATTERN = re.compile(r"<think>.*", flags=re.DOTALL)

# Thread lock for clean console output during parallel execution
print_lock = threading.Lock()


class GracefulKiller:
    """Handles Ctrl+C (SIGINT) and SIGTERM for clean shutdown."""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        if not self.kill_now:
            print("\n\n🛑 Ctrl+C detected! Cancelling pending tasks and shutting down...")
            self.kill_now = True
            # Raise KeyboardInterrupt to immediately break out of as_completed() or time.sleep()
            raise KeyboardInterrupt


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    if not text:
        return ""
    text = THINK_PATTERN.sub("", text)
    text = THINK_UNCLOSED_PATTERN.sub("", text)
    return text.strip()


def encode_image(image_path, max_size=1024):
    """Encode image to base64 with resizing and format conversion."""
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')

            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            mime_type = mimetypes.guess_type(image_path)[0] or 'image/jpeg'
            fmt = 'PNG' if mime_type == 'image/png' else 'JPEG'

            buffer = BytesIO()
            img.save(buffer, format=fmt, quality=85)
            return f"data:{mime_type};base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception:
        return None


def extract_caption(result: dict) -> str | None:
    """Extract the actual caption from an OpenAI-compatible response."""
    try:
        message = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None

    content = message.get("content") or ""
    caption = strip_thinking(content)

    if caption:
        return caption

    reasoning = message.get("reasoning_content") or ""
    if reasoning:
        return strip_thinking(reasoning)

    return None


def get_caption(image_url, api_base, model, api_token=None,
                prompt_text=DEFAULT_PROMPT, max_tokens=32768,
                disable_thinking=False):
    """Get image caption from API with error handling and retries."""
    endpoint = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    messages = []

    if disable_thinking:
        messages.append({
            "role": "system",
            "content": "/no_think"
        })

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    })

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    if disable_thinking:
        payload["options"] = {"think": False}
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=300
            )
            response.raise_for_status()
            result = response.json()

            caption = extract_caption(result)
            if caption:
                return caption

            return None

        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None


def process_image(img_path, caption_path, api_base, model, api_token, 
                  prompt_text, max_size, max_tokens, no_think, killer):
    """Worker function for parallel execution."""
    # Abort early if shutdown was requested
    if killer.kill_now:
        return False, img_path.name, "Cancelled"

    image_url = encode_image(img_path, max_size)
    if not image_url:
        return False, img_path.name, "Encode failed"

    caption = get_caption(
        image_url, api_base, model, api_token,
        prompt_text=prompt_text, max_tokens=max_tokens, disable_thinking=no_think
    )
    
    if not caption:
        return False, img_path.name, "API failed"

    try:
        with caption_path.open('w', encoding='utf-8') as f:
            f.write(caption)
        return True, img_path.name, caption_path.name
    except OSError as e:
        return False, img_path.name, str(e)


def main():
    parser = argparse.ArgumentParser(description='Batch Image Captioning (Parallel Supported)')
    parser.add_argument('input_dir', help='Image directory')
    parser.add_argument('--api_base', default='http://localhost:5000', help='API base URL')
    parser.add_argument('--api_token', help='API token for authentication')
    parser.add_argument('--model', default='gemma3', help='Model name')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='Custom captioning prompt')
    parser.add_argument('--max_size', type=int, default=1024, help='Max image dimension')
    parser.add_argument('--max_tokens', type=int, default=32768, help='Max tokens')
    parser.add_argument('--no_think', action='store_true', help='Disable thinking mode')
    parser.add_argument('--parallel', type=int, default=4, help='Number of concurrent API requests')
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_paths = []
    supported_exts = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp')
    for ext in supported_exts:
        image_paths.extend(Path(args.input_dir).glob(ext))

    to_process = []
    for img_path in image_paths:
        caption_path = Path(output_dir) / f"{img_path.stem}.txt"
        if not caption_path.exists():
            to_process.append((img_path, caption_path))

    total = len(to_process)
    if total == 0:
        print("No new images to process. All captions already exist.")
        return

    print("\n" + "=" * 80)
    print(f"🚀 Starting batch captioning for {total} images (Parallel: {args.parallel})")
    print(f"API Endpoint: {args.api_base}  |  Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}  |  Thinking: {'disabled' if args.no_think else 'enabled'}")
    print("=" * 80 + "\n")

    killer = GracefulKiller()
    start_time = time.time()
    success_count = 0
    fail_count = 0

    executor = ThreadPoolExecutor(max_workers=args.parallel)
    
    try:
        # Submit all tasks to the thread pool
        futures = {
            executor.submit(
                process_image, img_path, caption_path,
                args.api_base, args.model, args.api_token,
                args.prompt, args.max_size, args.max_tokens, args.no_think, killer
            ): (img_path, caption_path)
            for img_path, caption_path in to_process
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            if killer.kill_now:
                break
                
            img_path, caption_path = futures[future]
            try:
                success_flag, name, msg = future.result()
                if success_flag:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                
            # Update display
            completed = success_count + fail_count
            pct = (completed / total) * 100
            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            remaining = total - completed
            eta = avg_time * remaining
            speed = completed / elapsed if elapsed > 0 else 0.0
            
            with print_lock:
                status = (f"[{completed:03d}/{total:03d}] ({pct:5.1f}%) | "
                          f"Elapsed: {int(elapsed // 60)}m{int(elapsed % 60):02d}s | "
                          f"ETA: {int(eta // 60)}m{int(eta % 60):02d}s | "
                          f"Avg: {avg_time:.1f}s/img | Speed: {speed:.2f} img/s")
                # Pad with spaces to overwrite any previous longer text
                print(f"\r{status:<120}", end='', flush=True)
                
    except KeyboardInterrupt:
        # Clear the progress line and print interrupt message
        print(f"\r{' '*120}\r", end="")
        print("🛑 Interrupted by user! Cancelling pending tasks...")
        for f in futures:
            f.cancel()
    finally:
        # Ensure we are on a new line for the summary
        print()
        executor.shutdown(wait=False)

    total_time = time.time() - start_time
    cancelled_count = total - success_count - fail_count
    
    print(f"\n{'=' * 40} SUMMARY {'=' * 40}")
    print(f"Total queued:       {total}")
    print(f"Successful:         {success_count}")
    print(f"Failed:             {fail_count}")
    if cancelled_count > 0:
        print(f"Cancelled:          {cancelled_count}")
    print(f"Total time:         {int(total_time // 60)}m{int(total_time % 60):02d}s")
    if success_count > 0:
        print(f"Average time/image: {total_time / success_count:.1f}s (Total wall time)")
        print(f"Effective throughput: {success_count / total_time:.2f} images/sec")
    print("=" * 89)
    
    # Force exit to prevent Python's atexit handlers from waiting for blocked request threads
    if killer.kill_now:
        os._exit(0)


if __name__ == "__main__":
    main()