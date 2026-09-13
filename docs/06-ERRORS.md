# OTD 1.0 — Friendly Error Design

OTD errors never say "syntax error at line 3". They say **what you did, why it
can't work, and how to fix it** — with a suggestion engine that guesses your
typo.

## Error Anatomy

```
OTDError {
  line: 4,                        // editor jumps & underlines here
  message: "radius cannot be negative (-5)",
  hint: "did you mean 5cm?"       // shown in blue
}
```

## The Catalog

| # | You write | OTD answers |
|---|---|---|
| 1 | `sphere(r: -5cm)` | radius cannot be negative (-5) — did you mean 5cm? |
| 2 | `cube(s: -2cm)` | size cannot be negative (-2) — did you mean 2cm? |
| 3 | `sphre 2cm` | unknown shape 'sphre' — did you mean sphere? |
| 4 | `material: bronz` | unknown material 'bronz' — did you mean bronze? |
| 5 | `color: reed` | unknown color 'reed' — did you mean red? |
| 6 | `torus(r: 2cm, tube: -1mm)` | tube cannot be negative (-1) — did you mean 1mm? |
| 7 | `cup = cylinder(top: 4cm` | missing ')' — the '(' on line 3 was never closed |
| 8 | `at (1cm + 30deg, 0, 0)` | cannot add a length (1cm) and an angle (30deg) |
| 9 | `2cm * 3cm` | cannot multiply two lengths — did you want to scale something? |
| 10 | `at (i * 4cm, 0)` | 'at' needs 3 numbers — (x, y, z) |
| 11 | `use wheel …` (undefined) | 'wheel' is not defined yet — define it above this line |
| 12 | `handle = torus()` | …uses only defaults (this is fine!) — never an error |
| 13 | `repeat(n: 0)` | repeat count cannot be less than 1 — did you mean 1? |
| 14 | `divide by zero: h / 0` | division by zero — check the numbers above |
| 15 | `export pdf "x.pdf"` | I can export stl, obj, gltf, usdz, png or scad — not 'pdf' |
| 16 | `simulate: explode` | I can simulate drop, float or collapse — not 'explode' |
| 17 | `ask "mass"` (before any object) | nothing to weigh yet — add a shape first |
| 18 | `material: steel cup` | put a colon after the object name: material cup: steel |
| 19 | `1cm 2cm` (two numbers) | unexpected '2cm' — one value is enough here |
| 20 | `revolve []` | revolve needs profile points like [(2cm, 0), (4cm, 3cm)] |
| 21 | `loft [sphere 2cm]` | loft needs at least 2 cross-sections |
| 22 | future keyword (e.g. `snow`) | warning: 'snow' is not in OTD 1.0 — ignored (file still runs) |
| 23 | `cube 1cm at 2cm, 3cm, 0` | at wants a position in parentheses — like at (4cm, 5cm, 0) |
| 24 | `cube 1cm mirror xy` | mirror wants x, y, or z, not 'xy' — try mirror x |
| 25 | `cube(1cm))` | this ) doesn't match any ( — remove it, or open a ( earlier |
| 26 | `rotate 90 °` on its own | ° turns a number into degrees — put it right after the number, like 90° |
| 27 | `scene` (nothing after) | scene needs a title in quotes, like scene "Cup" |

## Suggestion Engine

- Vocabularies: 54 keywords · 27 materials · 147 colors · 6 formats · 3 sim modes.
- Match rule: Levenshtein distance ≤ 2, or the word starts with your typo.
- Category-aware: a typo near `material:` only searches materials; near `color:`
  only colors.
- All suggestions are **hints**, never silent auto-correction — the user stays
  in charge.

## Where Errors Appear

1. A red bar under the editor with the message + hint.
2. The offending line is underlined and the gutter shows a red marker.
3. The console repeats it with the line number.
4. Compilation stops at the first error (friendlier than a wall of red).
