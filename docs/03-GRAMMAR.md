# OTD 2.1 — Grammar (EBNF)

Notation: ISO/IEC 14977-style EBNF. `"…"` are literal terminals. Statements are
**line-based** — semicolons optional (`;` also separates, 2.1). A statement
continues onto the next line if the line ends with a comma or an unclosed
bracket. A **pattern's body may start on the next line** (indentation style),
as long as that line is not itself a new statement.

> **2.1 is the syntax-expansion release**: control flow (`if / else / end`),
> loops (`for / while / break / continue`), compound assignment (`+= -= *= /=`),
> comparisons and logic (`< > <= >= == != && || !` and `and or not is mod in`),
> power and remainder (`^` `%`), ranges (`1..12`), indexing and slicing
> (`xs[0]`, `xs[1..3]`), boolean literals (`true / false`), multi-statement
> `define … end` templates, `assert`, `print` interpolation, 17 new math
> functions, 4 new units (`km yd um rad`), scientific notation, and
> `#[ … ]#` block comments.
> **Every 1.0 and 2.0 file compiles unchanged** — the corpus is the gate.

## Lexical Rules

| Token | Form | Examples |
|---|---|---|
| comment | `#` to end of line — **unless it starts a `#rrggbb` color** | `# a coffee cup` |
| block comment | `#[` … `]#` — spans lines (2.1) | `#[ a long note ]#` |
| IDENT | letter { letter \| digit \| `_` } | `cup`, `gear_teeth` |
| NUMBER | digits [`.` digits] [exponent] with optional unit — glued (`10mm`) or **one space apart** (`10 mm`) | `42`, `3.5`, `1.5e3`, `2E-4`, `10mm`, `4 cm`, `2m`, `3in`, `6ft`, `45deg`, `90°`, `90 °` |
| STRING | `"…"` (curly quotes from pasted text are accepted) | `"Coffee Cup"` |
| operators | `=` `+` `-` `&` `*` `/` `(` `)` `[` `]` `,` `:` — plus the 2.1 set: `^` `%` `!` `<` `>` `<=` `>=` `==` `!=` `&&` `||` `..` `+=` `-=` `*=` `/=` `;` (fancy dashes from pasted text become `-`) | |
| colors | 147 names or `#rrggbb` (value position only) | `red`, `steelblue`, `#1e90ff` |
| materials | 27 names (value position only) | `steel`, `oak`, `ceramic` |

A bare number with no unit uses the scene default unit (`unit:`, default **cm**).
Keywords are not reserved in value positions, so `gold` can be a material *and*
a color — context decides. The 2.1 words (`if`, `for`, `and`, `true`, …) are
reserved everywhere: they can never be object names.

**`in` disambiguation (2.1):** attached `3in` is always three inches; a spaced
`3 in` becomes the membership operator when a value follows (`3 in [1, 2, 3]`,
`x in things`, `5 in 1..10`). Nobody writes inches followed by a list.

**Trailing commas** are legal before any closing bracket: `sphere(r: 2cm,)`,
`at (1cm, 2cm,)`, `[(0,0), (3cm,0),]`.

**Paste-proofing:** `–` `—` `−` are read as `-`; `“` `”` as `"`; a leading UTF-8
BOM is skipped. Pasting from web pages and word processors just works.

## Grammar

```ebnf
(* ============ OTD 1.0 — full grammar ============ *)

program   = { statement } ;

statement = [ stmt ] , NEWLINE ;
stmt      = comment
          | sceneStmt | versionStmt | unitStmt
          | defineStmt | useStmt | assignment | addStmt
          | applyStmt | hideStmt | showStmt | cameraStmt
          | gravityStmt | simulateStmt | askStmt | exportStmt
          | exprStmt ;

comment        = "#" , { character - NEWLINE } ;

sceneStmt      = "scene" , [ ":" ] , STRING ;
versionStmt    = "version" , [ ":" ] , INTEGER ;
unitStmt       = "unit" , ":" , LENGTH_UNIT ;
gravityStmt    = "gravity" , [ ":" ] , ( "off" | "earth" | "moon" | "mars" | NUMBER ) ;
cameraStmt     = "camera" , [ ":" ] , ( "front" | "top" | "side" | "iso" ) ;

assignment     = IDENT , "=" , expression ;
defineStmt     = "define" , [ ":" ] , IDENT , [ "(" , [ IDENT , { "," , IDENT } ] , ")" ] , "=" , expression ;
useStmt        = "use" , [ ":" ] , IDENT , [ "(" , argList , ")" ] , { postfix } ;
addStmt        = "add" , [ ":" ] , IDENT , "," , expression , { "," , expression } ;

applyStmt      = ( "material" | "color" ) , [ IDENT ] , ":" , value ;
hideStmt       = ( "hide" | "show" ) , [ ":" ] , IDENT ;

simulateStmt   = "simulate" , [ ":" ] , ( "drop" | "float" | "collapse" ) ;
askStmt        = "ask" , [ ":" ] , STRING ;
exportStmt     = "export" , [ ":" ] , ( "stl" | "obj" | "gltf" | "usdz" | "png" | "scad" ) , STRING ;
exprStmt       = expression ;                       (* anonymous object — shown *)

(* every statement tolerates the colon: OTD reads `print "hi"` and
   `print: "hi"` the same way — write whichever feels natural *)

(* ---------- expressions ---------- *)

expression     = boolean ;
boolean        = chain , { ( "+" | "-" | "&" ) , chain } ;      (* left-assoc *)
chain          = unary , { postfix } ;
unary          = [ "-" ] , primary ;
primary        = paren | tuple | list
                | patternCall | shapeCall | partCall | funcCall
                | IDENT | literal ;

paren          = "(" , expression , ")" ;
tuple          = "(" , expression , "," , expression , [ "," , expression ] , ")" ;
list           = "[" , [ exprList ] , "]" ;
exprList       = expression , { "," , expression } , [ "," ] ;       (* trailing comma allowed *)
literal        = NUMBER | STRING ;

(* a shape word with NO argument at all is a call with all defaults —
   the "everything predefined" promise: `cube`, `cup = sphere`,
   `cube at (5cm, 0, 0)`. A word you assigned (`tube = cube 1cm`) is
   always your variable, never a default shape. *)
shapeCall      = shapeWord , [ "(" , [ argList ] , ")"
                             | NUMBER                    (* sphere 5cm *)
                             | IDENT                     (* sphere ball_r — a variable as the size *)
                             ] ;
patternCall    = patternWord , "(" , [ argList ] , ")" , chain ;      (* inner copied n times —
                                                                     the chain may start on
                                                                     the NEXT line *)
partCall       = IDENT , "(" , [ argList ] , ")" ;
funcCall       = funcWord , "(" , exprList , ")" ;
argList        = arg , { "," , arg } , [ "," ] ;                 (* trailing comma allowed *)
arg            = IDENT , ":" , value | value ;      (* named or positional *)
(* `material:` / `color:` may appear INSIDE the argument list too:
   torus(r, 1cm, material: rubber), sphere(1cm, color: #1e90ff) *)

postfix        = "at" , tuple
                | "rotate" , ( ANGLE | tuple )
                | "scale" , ( NUMBER | tuple )
                | "mirror" , ( "x" | "y" | "z" )
                | ( "material" | "color" ) , ":" , value ;

shapeWord      = "sphere" | "cube" | "cylinder" | "cone" | "torus" | "pyramid"
                | "prism" | "capsule" | "wedge" | "plane" | "tube" | "helix"
                | "extrude" | "revolve" | "sweep" | "loft" | "text" | "import"
                | "terrain" | "metaball"
                | "hollow" | "group"
                | "add" | "subtract" | "intersect" ;
patternWord    = "repeat" | "grid" | "ring" | "scatter" ;
funcWord       = "cos" | "sin" | "sqrt" | "abs" | "min" | "max" | "round" ;

value          = literal | IDENT ;   (* resolved by context: material/color/on/off… *)
LENGTH_UNIT    = "mm" | "cm" | "m" | "in" | "ft" ;
```

## Operator Precedence (highest → lowest)

| Level | Operators | Associativity | Notes |
|---|---|---|---|
| 1 | postfix `at / rotate / scale / mirror / material: / color:` and indexing `xs[i]`, `xs[a..b]` | left-to-right chain | shapes |
| 2 | `^` | **right** | `-2 ^ 2` is `-(2²)`; plain numbers only |
| 3 | unary `-`, `!` / `not` | — | `not` binds looser than comparison: `not x == 5` means `not (x == 5)` |
| 4 | `*` `/` `%` / `mod` | left | dimension-checked `*` `/`; `%` plain numbers |
| 5 | `+` `-` `&` | left | booleans & arithmetic, same level |
| 6 | `<` `>` `<=` `>=` `==` `!=` `is` [`not`] `in` | left | answer `true`/`false`; units promote |
| 7 | `&&` / `and` | left | short-circuit |
| 8 | `\|\|` / `or` | left | short-circuit |
| 9 | `..` (range) | — | `a..b` inclusive, plain numbers |
| 10 | `=` `+=` `-=` `*=` `/=` (statement level) | — | |

Use parentheses when mixing: `(sphere 2cm + cube 3cm) at (5cm, 0, 0)`.

## Placement Semantics (the one rule to remember)

1. Every shape is built **resting on the ground** (y = 0).
2. `rotate` spins about the object's own center.
3. `at (x, y, z)` puts the object's **rest point** (bottom-center) at that spot.
4. Modifier chains apply left to right: `torus(2cm, 6mm) rotate (90deg, 0, 0) at (3cm, 2cm, 0)`
   = "spin it, then stand it there".

## Special Forms

- **Hollow sugar:** `a - hollow(wall: 3mm)` is shorthand for
  `a - inner(a, wall 3mm)` — i.e. `subtract(a, hollow(a, wall: 3mm))`.
- **Bare positional argument:** `sphere 5cm` = `sphere(r: 5cm)` (each shape has
  one "main" parameter).
- **`add cup, handle`** fuses handle into cup and removes the handle entry.
- **Patterns swallow the following chain:** in `repeat(n: 4) cube at (0, i*2cm, 0)`,
  the `at` belongs to each copy; `i` is the copy index. The chain may start on
  the next line — a following statement (an assignment or a statement keyword)
  is never mistaken for the body:
  ```
  fence = repeat(n: 5)
    cube 1cm material: wood
  next = cube 2cm          ← starts a new statement, not a 6th post
  ```
- **Loose named arguments accept arithmetic on numbers:**
  `depth: 5mm + 2mm` means depth 7mm. A following word (a shape) is never
  stolen: `hollow wall: 2mm + cube 1cm` unions the cube, it does not add the
  cube to the wall thickness.

---

## 2.0 Additions — four words, four productions (P0650–P1299)

```ebnf
(* ============ OTD 2.0 additions ============ *)

(* deep-tier postfix modifiers — chain exactly like at/rotate *)
postfixMod    = smoothMod | subdivMod ;

smoothMod     = "smooth" [ "(" smoothArgs ")" ] ;
smoothArgs    = { smoothArg , } ;
smoothArg     = ( "n" ":" expression ) | ( "strength" ":" expression ) | expression ;

subdivMod     = "subdiv" [ "(" [ "n" ":" ] expression [ "," ] ")" ] ;

(* the smooth-union builder — an ordinary call like hollow *)
blendCall     = "blend" "(" expression "," expression
                 [ "," blendArgs ] ")" ;
blendArgs     = "gap" ":" length | "res" ":" plain ;

(* the physics-solved cable — an ordinary primitive call *)
ropeCall      = "rope" "(" "from" ":" point "," "to" ":" point
                 [ "," ropeArgs ] ")" ;
ropeArgs      = "thickness" ":" length | "sag" ":" length ;
```

Defaults: `smooth(n: 1, strength: 0.5)` · `subdiv(n: 1)` · `gap: 5mm` ·
`res: 34` · `thickness: 4mm` · `sag: 6 % of span`.

## 2.0 Syntax Audit (the standing "find bugs and fix" gate)

Every release re-audits the grammar. Found and fixed in 2.0:

| # | Bug | Status |
|---|-----|--------|
| 1 | `metaball [(0,0,0), …]` — the documented call form never triggered call parsing (`takes_list` omitted metaball; the word became a bare ident) | **fixed** — metaball added to the list-taking call trigger |
| 2 | `pi` was documented as a predefined constant but never implemented — `pi` hit "I don't know what 'pi' is" | **fixed** — `pi` resolves in both the VM and the tree-walk |
| 3 | marching-tetrahedra tables: the 1.0 tet decomposition did not tile the cube (125 uncovered / 467 overlapping sample cells on the coverage test) — metaball meshes were never watertight | **fixed** — Freudenthal–Kuhn conforming triangulation; manifold-checked |
| 4 | 2-2 crossing case triangulated the isosurface quad with a shared *boundary* edge, leaving the diagonal uncovered — pinholes in every metaball/blend mesh | **fixed** — crossings walked as a cycle, split on a true diagonal |
| 5 | new words checked for collisions: `smooth` (was a terrain parameter — stays valid there by context), `blend`, `rope`, `subdiv` — no keyword/parameter/material/color collisions | **verified** |
| 6 | `ask "watertight?"` matched the `water` (float) rule first | **fixed** — watertight query checked before the buoyancy query |
| 7 | `rope(from: (0, 30cm, 0), …)` — single tuples were not accepted as paths (only lists were) | **fixed** — a lone tuple is a one-point path |
| 8 | all four new words appear only as calls/postfix — no new statement forms, no lexer changes; every 1.0 file compiles byte-identically (regression-gated) | **verified** |

The audit runs as tests: `cargo test --release` includes the coverage,
manifold, collision, and corpus-regression gates.

---

## 2.1 Additions — the syntax expansion

```ebnf
(* ============ OTD 2.1 additions ============ *)

(* ---- statements ---- *)

ifStmt     = "if" , expression , ifBody ;
ifBody     = ":" , stmt                        (* one-liner, no end *)
           | { statement } , ifTail ;
ifTail     = "end"
           | "else" , ifBody
           | "else" , "if" , expression , ifBody ;   (* else-if chains *)

forStmt    = "for" , IDENT , "=" , expression , "to" , expression
             [ "by" , expression ] , forBody ;
forInStmt  = "for" , IDENT , "in" , expression , forBody ;
forBody    = ":" , stmt | { statement } , "end" ;

whileStmt  = "while" , expression , forBody ;
breakStmt  = "break" ;
contStmt   = "continue" ;
assertStmt = "assert" , expression , [ "," , STRING ] ;

assignOpStmt = IDENT , ( "+=" | "-=" | "*=" | "/=" ) , expression ;

defineBlockStmt = "define" , [ ":" ] , IDENT , [ "(" , params , ")" ] ,
                  { statement } , "end" ;      (* the last line is the value *)

(* ---- expressions ---- *)

rangeExpr  = expression , ".." , expression ;   (* inclusive, plain numbers *)
indexExpr  = primary , "[" , ( expression | rangeExpr ) , "]" ;
boolLit    = "true" | "false" ;

(* ---- operator words ---- *)

logicOp    = "and" | "or" | "not" ;             (* == && || ! *)
cmpOp      = "is" [ "not" ]                     (* == != *)
           | "in" ;                              (* membership *)
modOp      = "mod" ;                             (* == % *)
```

### What the new constructs do (semantics)

**if** — conditions are `true`/`false` values, comparisons, or numbers
(nonzero = true; the words `on`/`off` work too). The `:` form runs ONE
statement on the same line; the block form runs until `end`. `else if`
chains as far as you like.

**for i = 1 to 10 [by 2]** — inclusive both ends. Steps may be plain numbers,
lengths (`for x = 0cm to 10cm by 2.5cm`) or angles (`for a = 0deg to 90deg by
15deg`). Counting down is automatic when `from > to` (`for i = 5 to 1` runs
5,4,3,2,1). The loop letter is local to the loop — it can't clobber a
same-named variable outside.

**for x in …** — walks a list (`for r in [1cm, 2cm, 3cm]`), a range
(`for i in 1..12`), a tuple, or a word's letters. Lists may hold shapes:
`for thing in [cube 1cm, sphere 1cm]`.

**while** — runs until the condition turns false; a 100,000-pass cap turns
runaways into a friendly error (`"this while never ends"`).

**break / continue** — leave / skip one pass of the innermost loop; a friendly
error outside loops.

**`x += 5cm`** — read, combine, write back. On objects it fuses:
`blob += cube 1cm at (10cm, 0, 0)` grows the named shape on stage.

**`^` and `%`/`mod`** — plain numbers only (squaring a length would be an
area; OTD keeps one length dimension — same rule as `*`). `2 ^ 3 ^ 2` is 512;
`-2 ^ 2` is -4; `7 mod 2` is 1.

**comparisons** — `< > <= >= == != is [not]`; lengths promote bare numbers
with the scene unit (`5cm > 40` is true under `unit: mm`). Mismatched
dimensions (`5cm > 90deg`) are a friendly error. `==` also compares words,
lists (element-wise) and ranges.

**`x in xs`** — membership in a list, substring in a word (`"cup" in
"cupcake"`), or range (`5 in 1..10`).

**ranges** — `1..12`, inclusive, plain numbers; walk them with for-in, slice
with them (`xs[1..3]`), test with `in`, or `sum(1..100)` them (5050).

**indexing** — `xs[0]` counts from zero; `xs[-1]` is the last; `xs[1..3]` is
an inclusive slice; positions and words index too (`p[0]`, `"cup"[0]` is
`"c"`).

**`define … end`** — a multi-statement template. Parameters bind, statements
run in a private scope (locals never leak, never reach the stage), and the
**last line that makes or names a shape is the value**:

```
define wheel(r)
  rim = torus(radius: r, tube: 1cm)
  hub = cylinder(r: r / 2, h: 2cm)
  rim + hub
end
use wheel(r: 3cm)
```

**assert** — `assert x > 0, "x must be positive"` fails loud with your
message; passing is silent.

**print interpolation** — `print "radius {r} done"` substitutes known names
(lengths print as mm, angles as deg); unknown `{names}` stay as written.

**new functions** — `floor ceil pow log ln exp sign hypot atan atan2 asin
acos` (inverse trig answers in degrees), `lerp(a, b, t)` and `clamp(x, lo,
hi)` carry units through, and `count sum avg` walk lists and ranges.

**new units** — `km` (1,000,000mm), `yd` (914.4mm), `um` (0.001mm), and
`rad` (57.2958deg) join mm cm m in ft deg.

**`;` separators** — `x = 1; y = 2` works like newlines (but not inside
brackets). **Scientific notation** — `1.5e3`, `2E-4`. **Block comments** —
`#[ … ]#` spans lines.

### 2.1 Syntax Audit (found and fixed during the expansion)

| # | Bug | Status |
|---|-----|--------|
| 1 | `3 in [1, 2, 3]` — the beginner unit-gluing (`3 in` = 3 inches) swallowed the membership operator; the list then indexed a number | **fixed** — spaced `in` before a value is the operator; attached `3in` stays inches |
| 2 | `5 in 1..10` — the `..` range was not parseable as the right side of `in` (or as a value at all) | **fixed** — `..` is a general expression operator |
| 3 | the OTD-ASM VM answered `inf` for `1 / 0` where the tree-walk errors | **fixed** — VM divide-by-zero now errors with the same words |
| 4 | VM errors were returned but never recorded — `x = <vm error>` vanished silently | **fixed** — every VM error is recorded before it propagates |
| 5 | loop bodies re-clone per pass, recycling AST addresses — the VM program cache could collide with a stale entry (a `cube(...)` evaluating to a number) | **fixed** — the cache clears each iteration; pattern bodies keep theirs |
| 6 | assignments inside `define … end` pushed their shapes onto the stage | **fixed** — template locals never reach the stage; the last line is the value |
| 7 | `cup += handle` updated the variable but not the named object already on stage | **fixed** — compound assignment grows the on-stage object |
| 8 | every 1.0 + 2.0 file compiles byte-identically (examples, 23 lessons, fuzz corpus) | **verified** — regression-gated

