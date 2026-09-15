//! P0120 — recursive-descent parser (OTD 2.1 syntax expansion). Line-based statements, loose named args,
//! bare positional args, patterns that swallow the following chain.
//!
//! 2.1 syntax expansion: `if / else / end` blocks (multi-arm, one-liner `:`
//! form), `for i = 1 to 10 by 2`, `for x in list-or-range`, `while`, `break` /
//! `continue`, compound assignment `+= -= *= /=`, multi-statement `define …
//! end` templates, `assert`, comparisons and logic (`< > <= >= == != && || !`
//! plus the words `and or not is is not in mod`), power `^`, modulo `%`,
//! ranges `a..b`, indexing / slicing `xs[i]` / `xs[a..b]`, `true / false`.
//!
//! Precedence, loosest to tightest:
//!   or  →  and  →  not  →  comparisons (== != < > <= >= is in)  →  + - &
//!   →  * / % mod  →  unary - !  →  ^  →  postfix (at/rotate/scale/mirror + [i])

use crate::lang::ast::*;
use crate::lang::keywords as kw;
use crate::lang::lexer::{lex, qty_of, Tok, Token};
use crate::units::Qty;

pub fn parse(src: &str, default_unit: &str) -> Program {
    let mut p = Parser::new(src, default_unit);
    let stmts = p.parse_program();
    Program { stmts, errors: p.errors }
}

struct Parser {
    t: Vec<Token>,
    pos: usize,
    unit: String,
    errors: Vec<crate::lang::errors::Error>,
    /// names assigned somewhere in the program (`tube = …`, `count += …`).
    /// A word that the user assigned is a VARIABLE, never a zero-arg shape
    /// call — mirrors the evaluator's env-first resolution.
    assigned: std::collections::HashSet<String>,
}

impl Parser {
    fn new(src: &str, default_unit: &str) -> Parser {
        let out = lex(src);
        // pre-scan: every `NAME = …` / `NAME += …` makes NAME a variable
        let mut assigned = std::collections::HashSet::new();
        for w in out.toks.windows(2) {
            if let (Tok::Ident(name), Tok::P(op)) = (&w[0].tok, &w[1].tok) {
                if matches!(*op, "=" | "+=" | "-=" | "*=" | "/=") {
                    assigned.insert(name.clone());
                }
            }
        }
        Parser { t: out.toks, pos: 0, unit: default_unit.to_string(), errors: out.errors, assigned }
    }

    // ---- token helpers ----
    // Every accessor clamps to the Eof token: a statement keyword at the end
    // of a file (no trailing newline) must produce a friendly error, never an
    // out-of-bounds panic.
    fn peek(&self) -> &Tok { &self.t[self.pos.min(self.t.len() - 1)].tok }
    fn peek_at(&self, off: usize) -> &Tok {
        let i = (self.pos + off).min(self.t.len() - 1);
        &self.t[i].tok
    }
    fn line(&self) -> usize { self.t[self.pos.min(self.t.len() - 1)].line }
    fn bump(&mut self) -> Tok {
        let i = self.pos.min(self.t.len() - 1);
        let t = self.t[i].tok.clone();
        if self.pos < self.t.len() - 1 { self.pos += 1; }
        t
    }
    fn is_p(&self, p: &str) -> bool { matches!(self.peek(), Tok::P(s) if *s == p) }
    fn eat_p(&mut self, p: &str) -> bool {
        if self.is_p(p) { self.pos += 1; true } else { false }
    }
    fn is_word(&self, w: &str) -> bool { matches!(self.peek(), Tok::Ident(s) if s == w) }
    fn eat_word(&mut self, w: &str) -> bool {
        if self.is_word(w) { self.pos += 1; true } else { false }
    }
    fn err(&mut self, msg: impl Into<String>) {
        self.errors.push(crate::lang::errors::Error::new(self.line(), msg));
    }
    fn err_hint(&mut self, msg: impl Into<String>, hint: impl Into<String>) {
        self.errors.push(crate::lang::errors::Error::new(self.line(), msg).with_hint(hint));
    }
    fn skip_nl(&mut self) {
        while matches!(self.peek(), Tok::Nl) { self.pos += 1; }
    }
    fn at_stmt_end(&self) -> bool {
        matches!(self.peek(), Tok::Nl | Tok::Eof)
    }

    // ---- program / statements ----
    fn parse_program(&mut self) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        self.skip_nl();
        while !matches!(self.peek(), Tok::Eof) {
            let before = self.pos;
            if let Some(s) = self.parse_stmt() {
                stmts.push(s);
            }
            if self.pos == before {
                // safety: never loop forever
                self.pos += 1;
            }
            self.skip_nl();
        }
        stmts
    }

    fn parse_stmt(&mut self) -> Option<Stmt> {
        let line = self.line();
        let word = match self.peek() {
            Tok::Ident(w) => w.clone(),
            _ => {
                // expression statement (e.g. starts with a paren or minus)
                let e = self.parse_expr();
                return Some(Stmt::ExprStmt(e, line));
            }
        };
        match word.as_str() {
            "scene" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Str(s) => Some(Stmt::Scene(s)),
                    _ => { self.err("scene needs a title in quotes, like scene \"Cup\""); None }
                }
            }
            "version" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Num(v, u) => {
                        if u.is_some() { self.err_hint("version takes a plain number", "try version 1"); }
                        Some(Stmt::Version(v))
                    }
                    _ => { self.err("version needs a number, like version 1"); None }
                }
            }
            "unit" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Ident(u) if kw::UNITS.contains(&u.as_str()) => Some(Stmt::Unit(u)),
                    Tok::Ident(u) => {
                        let s = crate::lang::errors::suggest(&u, kw::UNITS)
                            .map(|s| format!("did you mean {}?", s))
                            .unwrap_or_else(|| "units are mm, cm, m, in, ft".into());
                        self.err_hint(format!("'{}' is not a length unit", u), s);
                        None
                    }
                    _ => { self.err("unit needs a unit name, like unit: mm"); None }
                }
            }
            "gravity" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Ident(g) => Some(Stmt::Gravity(g)),
                    Tok::Num(v, _) => Some(Stmt::Gravity(format!("{}", v))),
                    _ => { self.err("gravity needs earth, moon, mars, off, or a number like 9.81"); None }
                }
            }
            // OTD3 P2110 — `temperature: 800` | `temperature: 350K` | `temperature: 72F`
            "temperature" => {
                self.bump();
                self.eat_p(":");
                let neg = if matches!(self.peek(), Tok::P("-")) { self.bump(); -1.0 } else { 1.0 };
                match self.bump() {
                    Tok::Num(v, unit) => {
                        // optional trailing unit identifier (K / F / C)
                        let mut spec = format!("{}{}", neg * v, unit.unwrap_or_default());
                        if let Tok::Ident(u) = self.peek().clone() {
                            if u.len() == 1 || u.eq_ignore_ascii_case("kelvin") || u.eq_ignore_ascii_case("fahrenheit") || u.eq_ignore_ascii_case("celsius") {
                                spec.push_str(&u);
                                self.bump();
                            }
                        }
                        Some(Stmt::Temperature(spec))
                    }
                    _ => { self.err("temperature needs a number like 800, 350K, or 72F"); None }
                }
            }
            "camera" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Ident(v) => Some(Stmt::Camera(v)),
                    _ => { self.err("camera wants a view: front, top, side, or iso"); None }
                }
            }
            "environment" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Ident(v) => {
                        if v == "density" {
                            // `environment: density 1200` — a custom medium
                            match self.bump() {
                                Tok::Num(n, _) => Some(Stmt::Environment(format!("{}", n))),
                                _ => { self.err("environment: density needs a number, like environment: density 1200"); None }
                            }
                        } else {
                            Some(Stmt::Environment(v))
                        }
                    }
                    Tok::Num(v, _) => Some(Stmt::Environment(format!("{}", v))),
                    _ => { self.err("environment needs a medium: air, vacuum, water, oil, or density <number>"); None }
                }
            }
            "mix" => {
                self.bump();
                self.eat_p(":");
                let a = match self.bump() {
                    Tok::Ident(a) => a,
                    _ => { self.err("mix needs two materials, like mix: water + oil"); None.unwrap_or_default() }
                };
                // optional '+' between them
                self.eat_p("+");
                let b = match self.bump() {
                    Tok::Ident(b) => b,
                    _ => { self.err("mix needs a second material after +, like mix: water + oil"); None.unwrap_or_default() }
                };
                Some(Stmt::Mix { a, b, line })
            }
            // OTD4 P2300 — `include "parts/wheel.otd" at (x, y, z)`
            "include" => {
                self.bump();
                self.eat_p(":");
                let file = match self.bump() {
                    Tok::Str(s) => s,
                    _ => {
                        self.err_hint(
                            "include needs a file path in quotes",
                            "like include: \"parts/wheel.otd\" at (10cm, 0, 0)",
                        );
                        return None;
                    }
                };
                // optional `at (x, y, z)` placement
                let mut at: Option<Vec<Expr>> = None;
                if let Tok::Ident(w) = self.peek().clone() {
                    if w == "at" {
                        self.bump();
                        if let Tok::P("(") = self.peek().clone() {
                            self.bump();
                            let mut items = Vec::new();
                            loop {
                                items.push(self.parse_expr());
                                if matches!(self.peek(), Tok::P(")")) { self.bump(); break; }
                                if !self.eat_p(",") { break; }
                            }
                            at = Some(items);
                        }
                    }
                }
                Some(Stmt::Include { file, at, line })
            }
            // OTD4 P2310 — `magnetize: name` or `magnetize: name moment: 0.5`
            "magnetize" => {
                self.bump();
                self.eat_p(":");
                let target = match self.bump() {
                    Tok::Ident(n) => n,
                    _ => {
                        self.err_hint(
                            "magnetize needs an object name",
                            "like magnetize: rotor  (then run simulate: magnet)",
                        );
                        return None;
                    }
                };
                // optional `moment: <number>` named argument
                let mut moment: Option<Expr> = None;
                if let Tok::Ident(w) = self.peek().clone() {
                    if w == "moment" {
                        self.bump();
                        self.eat_p(":");
                        moment = Some(self.parse_expr());
                    }
                }
                Some(Stmt::Magnetize { target, moment, line })
            }
            // OTD4 P2320 — `strict: overlap` | `strict: all` | `strict: off`
            "strict" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Ident(mode) => Some(Stmt::Strict(mode)),
                    _ => {
                        self.err_hint(
                            "strict needs a mode",
                            "try strict: overlap  (or strict: all, strict: off)",
                        );
                        None
                    }
                }
            }
            // OTD4 P2330 — `overlap` or `overlap: check`
            "overlap" => {
                self.bump();
                let mode = if self.eat_p(":") {
                    match self.bump() {
                        Tok::Ident(m) => m,
                        _ => "check".to_string(),
                    }
                } else {
                    "check".to_string()
                };
                Some(Stmt::Overlap { mode, line })
            }
            // OTD6 #5: connect: A B — electrical connectivity between parts
            "connect" => {
                self.bump();
                self.eat_p(":");
                let a = match self.bump() {
                    Tok::Ident(n) => n,
                    _ => { self.err_hint("connect needs two part names", "try connect: battery rotor"); None.unwrap_or_default() }
                };
                let b = match self.bump() {
                    Tok::Ident(n) => n,
                    _ => { self.err_hint("connect needs a second part name", "try connect: battery rotor"); None.unwrap_or_default() }
                };
                Some(Stmt::Connect { a, b, line })
            }
            "hide" | "show" => {
                let which = word == "show";
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Ident(name) => Some(if which { Stmt::Show(name) } else { Stmt::Hide(name) }),
                    _ => { self.err(format!("{} needs an object name, like {} legs", word, word)); None }
                }
            }
            "simulate" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Ident(sim) => {
                        if !["drop", "float", "collapse", "splash", "settle", "solidity", "gas", "mix",
                             "energy", "heat", "thermal", "thermo", "magnet", "magnetism",
                             "sound", "acoustics", "light", "optics", "time", "motion",
                             // OTD3.1 — the self-make expansion
                             "learn", "nn", "brain", "neural", "stats", "statistics",
                             "orbit", "astro", "space",
                             // OTD3.3 — the subatomic layer
                             "atom", "atoms", "nucleus", "nuclear",
                             "decay", "radioactive", "radioactivity", "halflife",
                             "particles", "particle", "standardmodel", "quark", "quarks",
                             // OTD4 — the dynamics expansion
                             "aero", "aerodynamics", "drag", "lift", "flight",
                             "fluid", "fluiddynamics", "fluid_dynamics", "bernoulli", "poiseuille",
                             "electro", "electrodynamics", "ohm", "current",
                             "stellar", "stellardynamics", "stellar_dynamics", "nbody", "virial",
                             "rigid", "rigidbody", "rigid_body", "inertia", "gyroscope", "spin",
                             "motor", "electric_motor", "electricmotor", "circuit"].contains(&sim.as_str()) {
                            self.err_hint(
                                format!("'{}' is not a simulation I know", sim),
                                "try drop, float, collapse, splash, settle, solidity, gas, mix, energy, heat, magnet, sound, light, time, learn, stats, orbit, atom, decay, particles, aero, fluid, electro, stellar, rigid, motor, or circuit",
                            );
                        }
                        Some(Stmt::Simulate(sim, line))
                    }
                    _ => { self.err("simulate needs drop, float, collapse, splash, settle, solidity, gas, mix, energy, heat, magnet, sound, light, time, learn, stats, orbit, atom, decay, particles, aero, fluid, electro, stellar, rigid, motor, or circuit"); None }
                }
            }
            // OTD3.3 P2250 — `particle: gluon` | `particle: proton` — one
            // particle's card from the Standard Model deck (fundamental or
            // composite): mass, charge, spin, and its one-paragraph story
            "particle" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Ident(name) => Some(Stmt::Particle(name)),
                    _ => { self.err("particle needs a name like particle: gluon, particle: proton, or particle: neutrino"); None }
                }
            }
            "ask" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Str(q) => Some(Stmt::Ask(q, line)),
                    _ => { self.err("ask needs a question in quotes, like ask \"mass?\""); None }
                }
            }
            "print" => {
                self.bump();
                self.eat_p(":");
                match self.bump() {
                    Tok::Str(s) => Some(Stmt::Print(s, line)),
                    _ => { self.err("print needs a message in quotes"); None }
                }
            }
            "export" => {
                self.bump();
                self.eat_p(":");
                let fmt = match self.bump() {
                    Tok::Ident(f) => f,
                    _ => { self.err("export needs a format: stl, obj, gltf, usdz, png, or scad"); return None }
                };
                let file = match self.bump() {
                    Tok::Str(f) => f,
                    _ => { self.err("export needs a file name in quotes, like export stl \"cup.stl\""); return None }
                };
                Some(Stmt::Export { fmt, file, line })
            }
            "define" => {
                self.bump();
                self.eat_p(":");
                let name = match self.bump() {
                    Tok::Ident(n) => n,
                    _ => { self.err("define needs a name, like define wheel = …"); return None }
                };
                let mut params = Vec::new();
                if self.eat_p("(") {
                    loop {
                        match self.bump() {
                            Tok::Ident(p) => params.push(p),
                            Tok::P(")") => break,
                            Tok::P(",") => {}
                            _ => { self.err("template parameters look like define gear(r) = …"); break }
                        }
                    }
                }
                if self.eat_p("=") {
                    let body = self.parse_expr();
                    Some(Stmt::Define { name, params, body, line })
                } else {
                    // block define (2.1): statements until `end`, last one is the value
                    let (stmts, _) = self.parse_block_until_end(line, "define");
                    if stmts.is_empty() {
                        self.err_hint(
                            format!("define {name} … end needs the shape it returns"),
                            "the last line before end is the value",
                        );
                    }
                    Some(Stmt::DefineBlock { name, params, stmts, line })
                }
            }
            "use" => {
                self.bump();
                self.eat_p(":");
                let name = match self.bump() {
                    Tok::Ident(n) => n,
                    _ => { self.err("use needs a template name, like use wheel"); return None }
                };
                let mut args = Vec::new();
                if self.is_p("(") {
                    args = self.parse_args();
                }
                let mods = self.parse_mods();
                Some(Stmt::Use { name, args, mods, line })
            }
            "add" => {
                self.bump();
                self.eat_p(":");
                let first = self.parse_expr();
                let mut rest = Vec::new();
                while self.eat_p(",") {
                    self.skip_nl();
                    if self.at_stmt_end() { break; } // trailing comma before end-of-statement
                    rest.push(self.parse_expr());
                }
                Some(Stmt::Add { first, rest, line })
            }
            "material" | "color" => {
                let kind = if word == "material" { ApplyKind::Material } else { ApplyKind::Color };
                self.bump();
                // material cup: ceramic (targeted)  OR  material: ceramic (scene default)
                let mut target: Option<String> = None;
                if matches!(self.peek(), Tok::Ident(_)) && matches!(self.peek_at(1), Tok::P(":")) {
                    if let Tok::Ident(t) = self.bump() {
                        target = Some(t);
                    }
                }
                if !self.eat_p(":") {
                    if self.at_stmt_end() {
                        self.err_hint(
                            format!("{} needs a value, like {}: steel", word, word),
                            "materials and colors are plain words, no quotes",
                        );
                    } else {
                        self.err(format!("{} wants a colon, like {} steel", word, word));
                    }
                    return None;
                }
                let value = match self.bump() {
                    Tok::Ident(v) => v,
                    _ => {
                        self.err_hint(
                            format!("{} needs a value, like {} ceramic", word, word),
                            "materials and colors are plain words, no quotes",
                        );
                        return None;
                    }
                };
                Some(Stmt::Apply { kind, target, value, line })
            }
            // ---- 2.1: control flow ----
            "if" => {
                self.bump();
                Some(self.parse_if(line))
            }
            "else" => {
                self.bump();
                self.err_hint(
                    "this else has no if",
                    "else goes after an if block: if … else … end",
                );
                self.consume_to_stmt_end();
                None
            }
            "end" => {
                self.bump();
                self.err_hint(
                    "this end doesn't close anything",
                    "it closes an if, for, while, or define block — is one missing its opening word?",
                );
                None
            }
            "while" => {
                self.bump();
                let cond = self.parse_expr();
                let body = self.parse_body(line, "while");
                Some(Stmt::While { cond, body, line })
            }
            "for" => {
                self.bump();
                let name = match self.bump() {
                    Tok::Ident(n) => n,
                    _ => {
                        self.err_hint(
                            "for needs a loop letter or name",
                            "like for i = 1 to 10, or for thing in stuff",
                        );
                        return None;
                    }
                };
                if self.eat_p("=") {
                    let from = self.parse_expr();
                    if !self.eat_word("to") {
                        self.err_hint(
                            "for counts from one value to another",
                            "like for i = 1 to 10 (by 2 to skip)",
                        );
                    }
                    let to = self.parse_expr();
                    let step = if self.eat_word("by") { Some(self.parse_expr()) } else { None };
                    let body = self.parse_body(line, "for");
                    Some(Stmt::For { var: name, from, to, step, body, line })
                } else if self.eat_word("in") {
                    let iter = self.parse_range_expr();
                    let body = self.parse_body(line, "for");
                    Some(Stmt::ForIn { var: name, iter, body, line })
                } else {
                    self.err_hint(
                        format!("for needs '=' or 'in' after {name}"),
                        "like for i = 1 to 10, or for thing in stuff",
                    );
                    None
                }
            }
            "break" => {
                self.bump();
                Some(Stmt::Break(line))
            }
            "continue" => {
                self.bump();
                Some(Stmt::Continue(line))
            }
            "assert" => {
                self.bump();
                let cond = self.parse_expr();
                let mut msg = None;
                if self.eat_p(",") {
                    match self.bump() {
                        Tok::Str(s) => msg = Some(s),
                        _ => self.err("the assert message goes in quotes, like assert x > 0, \"x must be positive\""),
                    }
                }
                Some(Stmt::Assert { cond, msg, line })
            }
            _ => {
                // typo'd statement keyword? "materal: steel" → did you mean material:?
                if let Tok::Ident(w) = self.peek().clone() {
                    if matches!(self.peek_at(1), Tok::P(":")) && !kw::is_keyword(&w) {
                        if let Some(s) = crate::lang::errors::suggest(&w, &[
                            "scene", "unit", "version", "gravity", "camera", "hide", "show",
                            "material", "color", "simulate", "ask", "export", "print", "define", "use",
                            "environment", "mix", "temperature", "particle",
                            "assert",
                        ]) {
                            self.err_hint(
                                format!("'{}' isn't a statement I know", w),
                                format!("did you mean {}: ?", s),
                            );
                            // consume to end of line to avoid cascades
                            self.consume_to_stmt_end();
                            return None;
                        }
                    }
                }
                // compound assignment (2.1): x += expr
                if let Tok::Ident(name) = self.peek().clone() {
                    let op = match self.peek_at(1) {
                        Tok::P("+=") => Some(BinOp::Add),
                        Tok::P("-=") => Some(BinOp::Sub),
                        Tok::P("*=") => Some(BinOp::Mul),
                        Tok::P("/=") => Some(BinOp::Div),
                        _ => None,
                    };
                    if let Some(op) = op {
                        self.bump(); // name
                        self.bump(); // op
                        let e = self.parse_expr();
                        return Some(Stmt::AssignOp { name, op, val: e, line });
                    }
                }
                // assignment or expression statement
                if let Tok::P("=") = self.peek_at(1) {
                    if let Tok::Ident(name) = self.bump() {
                        self.bump(); // =
                        let e = self.parse_expr();
                        return Some(Stmt::Assign(name, e, line));
                    }
                }
                let e = self.parse_expr();
                Some(Stmt::ExprStmt(e, line))
            }
        }
    }

    fn consume_to_stmt_end(&mut self) {
        while !self.at_stmt_end() && !matches!(self.peek(), Tok::Eof) {
            self.pos += 1;
        }
    }

    // ---- 2.1: block and inline bodies ----

    /// The body after `if cond` / `while cond` / `for …`: either a one-line
    /// statement after `:` or a multi-line block closed by `end`.
    fn parse_body(&mut self, open_line: usize, what: &str) -> Vec<Stmt> {
        if self.eat_p(":") {
            return self.parse_inline_body(what);
        }
        let (stmts, _) = self.parse_block_until_end(open_line, what);
        stmts
    }

    /// One statement on the same line after `:` — no `end` needed.
    fn parse_inline_body(&mut self, what: &str) -> Vec<Stmt> {
        if self.at_stmt_end() {
            self.err_hint(
                format!("the : after {what} is for one-liners"),
                "put the statement on the same line (if x > 5: cube 1cm), or drop the : and close the block with end",
            );
            return Vec::new();
        }
        match self.parse_stmt() {
            Some(s) => vec![s],
            None => Vec::new(),
        }
    }

    /// Statements until `end` (consumed) — or a friendly error at end of file.
    fn parse_block_until_end(&mut self, open_line: usize, what: &str) -> (Vec<Stmt>, BlockTerm) {
        let mut stmts = Vec::new();
        self.skip_nl();
        loop {
            match self.peek() {
                Tok::Eof => {
                    self.err_hint(
                        format!("this {what} needs its end"),
                        format!("add end on its own line to close the {what} block that starts on line {open_line}"),
                    );
                    return (stmts, BlockTerm::Eof);
                }
                Tok::Nl => { self.pos += 1; continue; }
                Tok::Ident(w) if w == "end" => { self.bump(); return (stmts, BlockTerm::End); }
                _ => {}
            }
            let before = self.pos;
            if let Some(s) = self.parse_stmt() {
                stmts.push(s);
            }
            if self.pos == before {
                self.pos += 1; // safety: never loop forever
            }
        }
    }

    /// Statements until `else` or `end` (the terminator is consumed).
    fn parse_block_until_else_or_end(&mut self, open_line: usize, what: &str) -> (Vec<Stmt>, BlockTerm) {
        let mut stmts = Vec::new();
        self.skip_nl();
        loop {
            match self.peek() {
                Tok::Eof => {
                    self.err_hint(
                        format!("this {what} needs its end"),
                        format!("add end on its own line to close the {what} block that starts on line {open_line}"),
                    );
                    return (stmts, BlockTerm::Eof);
                }
                Tok::Nl => { self.pos += 1; continue; }
                Tok::Ident(w) if w == "end" => { self.bump(); return (stmts, BlockTerm::End); }
                Tok::Ident(w) if w == "else" => { self.bump(); return (stmts, BlockTerm::Else); }
                _ => {}
            }
            let before = self.pos;
            if let Some(s) = self.parse_stmt() {
                stmts.push(s);
            }
            if self.pos == before {
                self.pos += 1;
            }
        }
    }

    /// `if` has been bumped. Parses arms + optional else, consuming the `end`.
    fn parse_if(&mut self, line: usize) -> Stmt {
        let mut arms: Vec<(Expr, Vec<Stmt>)> = Vec::new();
        let mut els: Option<Vec<Stmt>> = None;
        let mut cond = self.parse_expr();
        loop {
            // one-liner: if cond: stmt [else: stmt]
            if self.is_p(":") {
                self.bump();
                arms.push((cond, self.parse_inline_body("if")));
                if self.is_word("else") {
                    self.bump();
                    self.eat_p(":");
                    els = Some(self.parse_inline_body("else"));
                }
                break;
            }
            // block: statements until else / end
            let (then, term) = self.parse_block_until_else_or_end(line, "if");
            arms.push((cond, then));
            match term {
                BlockTerm::Else => {
                    if self.is_word("if") {
                        // else if — another arm
                        self.bump();
                        cond = self.parse_expr();
                        continue;
                    }
                    let (body, _) = self.parse_block_until_end(line, "if");
                    els = Some(body);
                    break;
                }
                BlockTerm::End => break,
                BlockTerm::Eof => break, // error already reported
            }
        }
        Stmt::If { arms, els, line }
    }

    // ---- expressions (2.1 precedence) ----

    /// Top level: a full expression, possibly a range `a..b` (2.1).
    fn parse_expr(&mut self) -> Expr {
        let lo = self.parse_or();
        if self.is_p("..") {
            self.bump();
            let hi = self.parse_or();
            Expr::Range(Box::new(lo), Box::new(hi))
        } else {
            lo
        }
    }

    /// `||` / or
    fn parse_or(&mut self) -> Expr {
        let mut lhs = self.parse_and();
        loop {
            let is_or = match self.peek() {
                Tok::P("||") => true,
                Tok::Ident(w) if w == "or" => true,
                _ => false,
            };
            if !is_or { break; }
            self.bump();
            self.skip_nl();
            let rhs = self.parse_and();
            lhs = Expr::Bin(BinOp::OrElse, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    /// `&&` / and
    fn parse_and(&mut self) -> Expr {
        let mut lhs = self.parse_not();
        loop {
            let is_and = match self.peek() {
                Tok::P("&&") => true,
                Tok::Ident(w) if w == "and" => true,
                _ => false,
            };
            if !is_and { break; }
            self.bump();
            self.skip_nl();
            let rhs = self.parse_not();
            lhs = Expr::Bin(BinOp::AndAlso, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    /// `!` / not (prefix)
    fn parse_not(&mut self) -> Expr {
        match self.peek() {
            Tok::P("!") => {
                self.bump();
                Expr::Not(Box::new(self.parse_not()))
            }
            Tok::Ident(w) if w == "not" => {
                self.bump();
                Expr::Not(Box::new(self.parse_not()))
            }
            _ => self.parse_cmp(),
        }
    }

    /// comparisons: == != < > <= >= is [not] in
    fn parse_cmp(&mut self) -> Expr {
        let mut lhs = self.parse_add();
        loop {
            let op: Option<BinOp> = match self.peek().clone() {
                Tok::P("==") => { self.bump(); Some(BinOp::Eq) }
                Tok::P("!=") => { self.bump(); Some(BinOp::Ne) }
                Tok::P("<") => { self.bump(); Some(BinOp::Lt) }
                Tok::P(">") => { self.bump(); Some(BinOp::Gt) }
                Tok::P("<=") => { self.bump(); Some(BinOp::Le) }
                Tok::P(">=") => { self.bump(); Some(BinOp::Ge) }
                Tok::Ident(ref w) if w == "in" => { self.bump(); Some(BinOp::In) }
                Tok::Ident(ref w) if w == "is" => {
                    self.bump();
                    let neg = self.eat_word("not");
                    Some(if neg { BinOp::Ne } else { BinOp::Eq })
                }
                _ => None,
            };
            let op = match op { Some(o) => o, None => break };
            // `in` may take a range: x in 1..10
            let rhs = if op == BinOp::In { self.parse_expr() } else { self.parse_add() };
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    /// Level: + - & (and words add/subtract/intersect handled as calls)
    fn parse_add(&mut self) -> Expr {
        let mut lhs = self.parse_term();
        loop {
            let op = match self.peek() {
                Tok::P("+") => BinOp::Add,
                Tok::P("-") => BinOp::Sub,
                Tok::P("&") => BinOp::And,
                _ => break,
            };
            self.bump();
            self.skip_nl();
            let rhs = self.parse_term();
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    /// Level: * / % mod
    fn parse_term(&mut self) -> Expr {
        let mut lhs = self.parse_unary();
        loop {
            let op = match self.peek() {
                Tok::P("*") => BinOp::Mul,
                Tok::P("/") => BinOp::Div,
                Tok::P("%") => BinOp::Mod,
                Tok::Ident(w) if w == "mod" => BinOp::Mod,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_unary();
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    /// unary minus + power
    fn parse_unary(&mut self) -> Expr {
        if self.is_p("-") {
            self.bump();
            return Expr::Neg(Box::new(self.parse_unary()));
        }
        self.parse_pow()
    }

    /// `^` — right-associative, binds tighter than unary minus (-2^2 = -(2^2))
    fn parse_pow(&mut self) -> Expr {
        let base = self.parse_postfix();
        if self.is_p("^") {
            self.bump();
            let rhs = self.parse_unary();
            Expr::Bin(BinOp::Pow, Box::new(base), Box::new(rhs))
        } else {
            base
        }
    }

    /// postfix modifiers + indexing; patterns swallow the following chain
    fn parse_postfix(&mut self) -> Expr {
        let mut base = self.parse_primary();
        if matches!(base, Expr::Call { ref name, .. } if kw::is_pattern(name)) {
            return base;
        }
        loop {
            let mods = self.parse_mods();
            if !mods.is_empty() {
                base = Expr::Postfix { base: Box::new(base), mods };
                continue;
            }
            // indexing / slicing (2.1): xs[0], xs[1..3]
            if self.is_p("[") {
                self.bump();
                self.skip_nl();
                if self.eat_p("]") {
                    self.err("an index needs a number, like xs[0]");
                    continue;
                }
                let idx = self.parse_range_expr();
                self.skip_nl();
                if !self.eat_p("]") {
                    self.err("an index ends with ] — did you forget one?");
                }
                base = Expr::Index(Box::new(base), Box::new(idx));
                continue;
            }
            break;
        }
        base
    }

    /// An expression that may be `a..b` — same as parse_expr (kept for clarity
    /// at the for-in / index call sites).
    fn parse_range_expr(&mut self) -> Expr {
        self.parse_expr()
    }

    /// Parse postfix modifiers: at / rotate / scale / mirror / material: / color:
    fn parse_mods(&mut self) -> Vec<Mod> {
        let mut mods = Vec::new();
        loop {
            if self.is_word("at") {
                self.bump();
                if self.is_p("(") {
                    let tup = self.parse_tuple();
                    mods.push(Mod::At(tup));
                } else {
                    // beginner form: `at 4cm, 5cm, 0` — recover with a hint
                    self.err_hint(
                        "at wants a position in parentheses",
                        "like at (4cm, 5cm, 0)",
                    );
                    let mut items = vec![self.parse_expr()];
                    while self.eat_p(",") {
                        self.skip_nl();
                        if self.at_stmt_end() { break; }
                        items.push(self.parse_expr());
                    }
                    mods.push(Mod::At(items));
                }
            } else if self.is_word("rotate") {
                self.bump();
                let v = self.parse_angle_or_tuple();
                // OTD4 — optional `pivot (x, y, z)` | `pivot: origin` | `pivot: center`
                if self.is_word("pivot") {
                    self.bump();
                    let pivot = if self.is_p("(") {
                        PivotSpec::Point(self.parse_tuple())
                    } else if self.is_word("origin") {
                        self.bump();
                        PivotSpec::Origin
                    } else if self.is_word("center") {
                        self.bump();
                        PivotSpec::Center
                    } else if self.eat_p(":") {
                        if self.is_word("origin") {
                            self.bump();
                            PivotSpec::Origin
                        } else if self.is_word("center") {
                            self.bump();
                            PivotSpec::Center
                        } else if self.is_p("(") {
                            PivotSpec::Point(self.parse_tuple())
                        } else {
                            self.err_hint(
                                "pivot wants origin, center, or (x, y, z)",
                                "like rotate (0, 45deg, 0) pivot (0, 0, 0)",
                            );
                            PivotSpec::Center
                        }
                    } else {
                        PivotSpec::Center
                    };
                    mods.push(Mod::RotateWithPivot { angles: v, pivot });
                } else {
                    mods.push(Mod::Rotate(v));
                }
            } else if self.is_word("scale") {
                self.bump();
                let v = self.parse_factor_or_tuple();
                mods.push(Mod::Scale(v));
            } else if self.is_word("mirror")
                && matches!(self.peek_at(1), Tok::Ident(_))
            {
                self.bump();
                if let Tok::Ident(axis) = self.bump() {
                    let ax = axis.chars().next().unwrap_or('x');
                    if axis.chars().count() != 1 || !matches!(ax, 'x' | 'y' | 'z') {
                        self.err_hint(format!("mirror wants x, y, or z, not '{}'", axis), "try mirror x");
                    }
                    mods.push(Mod::Mirror(ax));
                }
            } else if self.is_word("smooth") {
                self.bump();
                // smooth  ·  smooth(3)  ·  smooth(n: 3)  ·  smooth(n: 3, strength: 0.5)
                let (mut it, mut st) = (None, None);
                if self.eat_p("(") {
                    loop {
                        if self.is_word("n") && matches!(self.peek_at(1), Tok::P(":")) {
                            self.bump();
                            self.bump();
                            it = Some(self.parse_expr());
                        } else if self.is_word("strength") && matches!(self.peek_at(1), Tok::P(":")) {
                            self.bump();
                            self.bump();
                            st = Some(self.parse_expr());
                        } else {
                            let v = self.parse_expr();
                            if it.is_none() { it = Some(v) } else if st.is_none() { st = Some(v) }
                        }
                        if !self.eat_p(",") { break; }
                        self.skip_nl();
                    }
                    self.eat_p(")");
                }
                mods.push(Mod::Smooth {
                    iterations: it.unwrap_or(Expr::Num(Qty::plain(1.0))),
                    strength: st.unwrap_or(Expr::Num(Qty::plain(0.5))),
                });
            } else if self.is_word("subdiv") {
                self.bump();
                let mut it = None;
                if self.eat_p("(") {
                    loop {
                        if self.is_word("n") && matches!(self.peek_at(1), Tok::P(":")) {
                            self.bump();
                            self.bump();
                            it = Some(self.parse_expr());
                        } else {
                            it = Some(self.parse_expr());
                        }
                        if !self.eat_p(",") { break; }
                        self.skip_nl();
                    }
                    self.eat_p(")");
                }
                mods.push(Mod::Subdiv(it.unwrap_or(Expr::Num(Qty::plain(1.0)))));
            } else if (self.is_word("material") || self.is_word("color")) && matches!(self.peek_at(1), Tok::P(":")) {
                let which = if let Tok::Ident(w) = self.peek().clone() { w } else { unreachable!() };
                self.bump();
                self.bump(); // :
                if let Tok::Ident(v) = self.bump() {
                    if which == "material" { mods.push(Mod::Material(v)) } else { mods.push(Mod::Color(v)) }
                } else {
                    self.err(format!("{} needs a plain word value, like {} gold", which, which));
                }
            } else {
                break;
            }
        }
        mods
    }

    /// ( a, b ) or ( a, b, c ) — after the opening paren is consumed by caller
    fn parse_tuple_inner(&mut self) -> Vec<Expr> {
        let mut items = vec![self.parse_expr()];
        while self.eat_p(",") {
            self.skip_nl();
            if self.is_p(")") { break; } // trailing comma is fine
            items.push(self.parse_expr());
        }
        if !self.eat_p(")") {
            self.err("a tuple ends with ) — did you forget one?");
        }
        if items.len() > 3 {
            self.err_hint(
                "tuples hold 2 or 3 values",
                "positions are (x, y, z); profile points are (radius, height)",
            );
        }
        items
    }

    fn parse_tuple(&mut self) -> Vec<Expr> {
        // assumes current token is '('
        self.bump();
        self.parse_tuple_inner()
    }

    fn parse_angle_or_tuple(&mut self) -> Expr {
        if self.is_p("(") {
            Expr::Tuple(self.parse_tuple())
        } else {
            // could be a bare number or expression
            self.parse_term_no_chain_mods()
        }
    }

    fn parse_factor_or_tuple(&mut self) -> Expr {
        if self.is_p("(") {
            Expr::Tuple(self.parse_tuple())
        } else {
            self.parse_term_no_chain_mods()
        }
    }

    /// A term that stops before postfix modifiers (for rotate/scale bare args).
    fn parse_term_no_chain_mods(&mut self) -> Expr {
        let mut lhs = if self.is_p("-") {
            self.bump();
            Expr::Neg(Box::new(self.parse_primary()))
        } else {
            self.parse_primary()
        };
        loop {
            let op = match self.peek() {
                Tok::P("*") => BinOp::Mul,
                Tok::P("/") => BinOp::Div,
                _ => break,
            };
            self.bump();
            let rhs = if self.is_p("-") {
                self.bump();
                Expr::Neg(Box::new(self.parse_primary()))
            } else {
                self.parse_primary()
            };
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_primary(&mut self) -> Expr {
        let line = self.line();
        match self.peek().clone() {
            Tok::Num(v, ref unit) => {
                let (v, unit) = (v, unit.clone());
                self.bump();
                let q = qty_of(v, &unit, line, &self.unit).unwrap_or(crate::units::Qty::plain(0.0));
                Expr::Num(q)
            }
            Tok::Str(s) => {
                self.bump();
                Expr::Str(s)
            }
            Tok::Ident(w) => {
                // boolean literals (2.1)
                if w == "true" {
                    self.bump();
                    return Expr::Bool(true);
                }
                if w == "false" {
                    self.bump();
                    return Expr::Bool(false);
                }
                // Decide if this word is a CALL: any ident + "(" (shape, func, or
                // template); primitive + bare number; extrude/revolve/sweep + bare list.
                let next_is_paren = matches!(self.peek_at(1), Tok::P("("));
                let next_is_num = matches!(self.peek_at(1), Tok::Num(..));
                let next_is_list = matches!(self.peek_at(1), Tok::P("["));
                let next_is_str = matches!(self.peek_at(1), Tok::Str(_));
                // `sphere -5cm` — negative bare argument (friendly error path).
                // (the ident is NOT consumed yet: look 1 and 2 tokens ahead)
                let next_is_neg = matches!(self.peek_at(1), Tok::P("-"))
                    && matches!(self.peek_at(2), Tok::Num(..));
                // metaball's documented form is `metaball [(x,y,z)…]` — it
                // was missing from this list in 1.0 (the call never fired;
                // found by the 2.0 syntax audit)
                let takes_list = w == "extrude" || w == "revolve" || w == "sweep" || w == "metaball";
                let takes_str = w == "text" || w == "import";
                // ZERO-ARG smart defaults ("everything predefined"): a bare shape
                // word with nothing that could be an argument is a call with all
                // defaults — `cube` alone, `cup = sphere`, `cube at (5cm, 0, 0)`.
                // Trigger only on unambiguous boundaries so variables named like
                // shapes still resolve in comma lists (`add tube, handle`).
                let is_shape_word = (kw::is_primitive(&w) || kw::is_builder(&w) || kw::is_pattern(&w))
                    && !self.assigned.contains(&w);
                let next_starts_loose_arg = match self.peek_at(1) {
                    // `sphere ball_r` (variable as the main size) or
                    // `ring radius: 5cm cube 1cm` / `hollow wall: 3mm` (loose named).
                    // Keywords (at/rotate/… and the 2.1 words if/else/end/and/or/…)
                    // are boundaries, never arguments.
                    Tok::Ident(m) if !kw::is_keyword(m) => {
                        !self.is_p_at(2, "(") // `sphere gear(...)` is not an argument
                    }
                    _ => false,
                };
                let zero_arg_call = is_shape_word && match self.peek_at(1) {
                    Tok::Nl | Tok::Eof => true,
                    Tok::P(")") | Tok::P("]") => true,
                    Tok::Ident(m) => match m.as_str() {
                        "at" | "rotate" | "scale" | "mirror" => true,
                        "material" | "color" => self.is_p_at(2, ":"),
                        _ => next_starts_loose_arg,
                    },
                    _ => false,
                };
                if next_is_paren
                    || (next_is_num && kw::is_primitive(&w))
                    || (next_is_neg && kw::is_primitive(&w))
                    || (next_is_list && takes_list)
                    || (next_is_str && takes_str)
                    || zero_arg_call
                {
                    self.bump(); // consume the word
                    return self.parse_call(w, line);
                }
                self.bump();
                // a 2.1 statement word misplaced mid-expression → explain it
                if matches!(w.as_str(),
                    "if" | "else" | "end" | "while" | "break" | "continue" | "assert"
                    | "and" | "or" | "not" | "is" | "mod" | "to" | "by" | "in" | "for") {
                    let hint = match w.as_str() {
                        "end" => "end closes an if, for, while, or define block — it goes on its own line".to_string(),
                        "if" | "while" | "for" => format!("{w} starts its own statement — write it on a new line"),
                        "and" | "or" => format!("{w} joins two comparisons — like x > 1 {w} x < 5"),
                        "not" => "not flips a comparison — like not x > 5".into(),
                        "is" => "is compares two things — like x is 5 (or is not)".into(),
                        "mod" => "mod takes the remainder — like 7 mod 2".into(),
                        "to" | "by" => format!("{w} belongs to a for line — like for i = 1 to 10"),
                        _ => format!("{w} is a keyword — it can't be used as a value here"),
                    };
                    self.err_hint(format!("'{w}' can't be used as a value here"), hint);
                }
                Expr::Ident(w)
            }
            Tok::P("(") => {
                self.bump();
                let first = self.parse_expr();
                if self.is_p(",") {
                    // tuple
                    let mut items = vec![first];
                    while self.eat_p(",") {
                        self.skip_nl();
                        if self.is_p(")") { break; } // trailing comma is fine
                        items.push(self.parse_expr());
                    }
                    if !self.eat_p(")") {
                        self.err("a tuple ends with ) — did you forget one?");
                    }
                    if items.len() > 3 {
                        self.err_hint(
                            "tuples hold 2 or 3 values",
                            "positions are (x, y, z); profile points are (radius, height)",
                        );
                    }
                    Expr::Tuple(items)
                } else {
                    if !self.eat_p(")") {
                        self.err("an opened ( needs its closing )");
                    }
                    first
                }
            }
            Tok::P("[") => {
                self.bump();
                let mut items = Vec::new();
                self.skip_nl();
                if !self.is_p("]") {
                    loop {
                        items.push(self.parse_expr());
                        self.skip_nl();
                        if self.eat_p(",") {
                            self.skip_nl();
                            if self.is_p("]") { break; } // trailing comma is fine
                            continue;
                        }
                        break;
                    }
                }
                if !self.eat_p("]") {
                    self.err("a list ends with ] — did you forget one?");
                }
                Expr::List(items)
            }
            other => {
                self.err_hint(
                    format!("I expected a shape or a value here, not {}", tok_desc(&other)),
                    "try a shape like sphere 2cm, or a name like cup",
                );
                self.bump();
                Expr::Ident(String::new())
            }
        }
    }

    /// Parse a call: name has been bumped. Handles:
    ///   name(args…)          — parens form
    ///   name 5cm | name [..] — bare positional arg
    ///   name … looseNamed: v — loose named args (e.g. extrude […] depth: 5mm)
    /// Patterns then swallow the following chain as `body`.
    fn parse_call(&mut self, name: String, line: usize) -> Expr {
        let mut args: Vec<Arg> = Vec::new();
        if self.is_p("(") {
            args = self.parse_args();
        } else if self.is_p("-") && matches!(self.peek_at(1), Tok::Num(..)) {
            // bare negative number (sphere -5cm → friendly negative-size error)
            if let Tok::Num(v, ref unit) = self.peek_at(1).clone() {
                let unit = unit.clone();
                self.bump(); // -
                self.bump(); // number
                if let Ok(q) = qty_of(v, &unit, line, &self.unit) {
                    args.push(Arg { name: None, val: Expr::Num(Qty { v: -q.v, dim: q.dim }) });
                }
            }
        } else if let Tok::Num(v, ref unit) = self.peek().clone() {
            // bare number arg (sphere 5cm)
            let unit = unit.clone();
            self.bump();
            if let Ok(q) = qty_of(v, &unit, line, &self.unit) {
                args.push(Arg { name: None, val: Expr::Num(q) });
            } else {
                args.push(Arg { name: None, val: Expr::Num(crate::units::Qty::plain(v)) });
                if let Some(e) = qty_of(v, &unit, line, &self.unit).err() { self.errors.push(e); }
            }
        } else if let Tok::Ident(v) = self.peek().clone() {
            // bare variable argument — `sphere ball_r`, `cube size` (primitives only;
            // patterns/builders take named or parenthesized args)
            if kw::is_primitive(&name)
                && !kw::is_keyword(&v)
                && !self.is_p_at(1, "(")
                && !self.is_p_at(1, ":")
            {
                self.bump();
                args.push(Arg { name: None, val: Expr::Ident(v) });
            }
        } else if self.is_p("[") {
            // bare list arg (extrude [(0,0), …], revolve […])
            let list = self.parse_primary(); // parses the [ … ] as List
            args.push(Arg { name: None, val: list });
        } else if let Tok::Str(s) = self.peek().clone() {
            // bare string arg (text "HELLO")
            self.bump();
            args.push(Arg { name: None, val: Expr::Str(s) });
        }
        // loose named args: IDENT ':' value (but material:/color: are postfix)
        loop {
            if let Tok::Ident(key) = self.peek().clone() {
                if (key == "material" || key == "color") && matches!(self.peek_at(1), Tok::P(":")) {
                    break; // postfix modifier, belongs to the chain
                }
                if matches!(self.peek_at(1), Tok::P(":")) && !kw::is_keyword(&key) {
                    // looks like a named parameter of this call
                    if self.loose_arg_allowed(&name, &key) {
                        self.bump(); // key
                        self.bump(); // :
                        let val = self.parse_loose_value();
                        args.push(Arg { name: Some(key), val });
                        continue;
                    }
                }
            }
            break;
        }
        // patterns swallow the following chain — the body may also start on
        // the NEXT line (natural multi-line style):
        //   fence = repeat(n: 5)
        //     cube 1cm
        let mut body: Option<Box<Expr>> = None;
        if kw::is_pattern(&name) {
            if self.body_follows() {
                let inner = self.parse_pattern_body();
                body = Some(Box::new(inner));
            }
        }
        Expr::Call { name, line, args, body }
    }

    /// After a pattern's arguments, decide whether the shape to repeat starts
    /// on this line or a following one — WITHOUT swallowing the next statement
    /// when the pattern simply has no body (e.g. the next line is an assignment
    /// or another statement keyword).
    fn body_follows(&mut self) -> bool {
        let save = self.pos;
        self.skip_nl();
        let ok = match self.peek() {
            Tok::Eof | Tok::Nl => false,
            Tok::Num(..) | Tok::Str(_) => true,
            Tok::P("-") | Tok::P("(") | Tok::P("[") => true,
            Tok::P(_) => false,
            Tok::Ident(w) => {
                let is_stmt_word = matches!(
                    w.as_str(),
                    "scene" | "version" | "unit" | "gravity" | "camera" | "hide" | "show"
                        | "define" | "use" | "add" | "material" | "color" | "simulate"
                        | "environment" | "mix" | "particle"
                        | "ask" | "export" | "print"
                        // 2.1 statements — never pattern bodies
                        | "if" | "else" | "end" | "for" | "while" | "break" | "continue" | "assert"
                );
                if is_stmt_word { false } else { !self.is_p_at(1, "=") }
            }
        };
        if !ok { self.pos = save; }
        ok
    }

    fn is_p_at(&self, off: usize, p: &str) -> bool {
        matches!(self.peek_at(off), Tok::P(s) if *s == p)
    }

    /// Value of a loose named argument: arithmetic with + - is allowed when
    /// the right side is a NUMBER literal (`depth: 5 + 2mm`). A right side
    /// starting with a word (a shape, e.g. `hollow(wall: 2mm) + cube 1cm`)
    /// belongs to the outer boolean expression, so it is never stolen.
    fn parse_loose_value(&mut self) -> Expr {
        let mut lhs = self.parse_term_no_chain_mods();
        loop {
            let op = match self.peek() {
                Tok::P("+") => BinOp::Add,
                Tok::P("-") => BinOp::Sub,
                _ => break,
            };
            if !matches!(self.peek_at(1), Tok::Num(..)) { break; }
            self.bump();
            let rhs = self.parse_term_no_chain_mods();
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_pattern_body(&mut self) -> Expr {
        // the chain consumed by a pattern: unary + postfix (incl. indexing)
        if self.is_p("-") {
            self.bump();
            return Expr::Neg(Box::new(self.parse_pattern_body()));
        }
        self.parse_postfix()
    }

    fn loose_arg_allowed(&self, shape: &str, key: &str) -> bool {
        // Parameter names for shapes that support loose named args.
        const PARAMS: &[(&str, &[&str])] = &[
            ("extrude", &["depth", "smooth"]),
            ("revolve", &["angle", "smooth"]),
            ("hollow", &["wall", "open"]),
            ("sweep", &["twist"]),
            ("loft", &["smooth"]),
            ("text", &["depth", "size"]),
            ("terrain", &["size", "height", "seed", "smooth"]),
            ("metaball", &["r", "strength", "seed"]),
            ("import", &["center"]),
            ("repeat", &["n", "step"]),
            ("grid", &["nx", "nz", "spacing"]),
            ("ring", &["n", "radius", "from", "to", "axis"]),
            ("scatter", &["n", "radius", "seed"]),
            ("tube", &["r", "path"]),
            ("helix", &["radius", "pitch", "turns", "tube"]),
            ("sphere", &["r", "radius", "smooth"]),
            ("cube", &["s", "size", "w", "d", "h"]),
            ("cylinder", &["r", "radius", "h", "height", "top", "bottom", "smooth"]),
            ("cone", &["r", "radius", "h", "height", "smooth"]),
            ("torus", &["radius", "r", "tube", "smooth"]),
            ("pyramid", &["base", "h", "sides"]),
            ("prism", &["sides", "r", "radius", "h"]),
            ("capsule", &["r", "radius", "h"]),
            ("wedge", &["s", "size", "w", "d", "h"]),
            ("plane", &["s", "size", "w", "d", "h"]),
            ("rope", &["from", "to", "thickness", "sag"]),
            ("blend", &["gap"]),
        ];
        for (s, keys) in PARAMS {
            if *s == shape {
                return keys.contains(&key);
            }
        }
        false
    }

    fn parse_args(&mut self) -> Vec<Arg> {
        // current token is '('
        self.bump();
        let mut args = Vec::new();
        self.skip_nl();
        if self.is_p(")") {
            self.bump();
            return args;
        }
        loop {
            // Inside parens, IDENT ':' is always a named argument
            // (positional expressions can never be followed by ':').
            // Note: this is why param names may coincide with keywords
            // (e.g. torus's `tube:` vs the tube primitive) — context decides.
            let named = match self.peek() {
                Tok::Ident(key) if matches!(self.peek_at(1), Tok::P(":")) => Some(key.clone()),
                _ => None,
            };
            if let Some(key) = named {
                self.bump(); // key
                self.bump(); // :
                let val = self.parse_expr();
                args.push(Arg { name: Some(key), val });
            } else {
                let val = self.parse_expr();
                args.push(Arg { name: None, val });
            }
            self.skip_nl();
            if self.eat_p(",") {
                self.skip_nl();
                if self.is_p(")") { self.bump(); break; } // trailing comma before ) is fine
                continue;
            }
            if self.eat_p(")") {
                break;
            }
            if self.at_stmt_end() {
                self.err("an argument list ends with ) — did you forget one?");
                break;
            }
        }
        args
    }
}

/// How a block body ended (2.1).
enum BlockTerm {
    End,
    Else,
    Eof,
}

fn tok_desc(t: &Tok) -> String {
    match t {
        Tok::Nl => "the end of the line".into(),
        Tok::Ident(s) => format!("'{}'", s),
        Tok::Num(v, _) => format!("the number {}", v),
        Tok::Str(s) => format!("\"{}\"", s),
        Tok::P(p) => format!("'{}'", p),
        Tok::Eof => "the end of the file".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(src: &str) -> Program {
        let p = parse(src, "cm");
        assert!(p.errors.is_empty(), "parse errors: {:#?}", p.errors);
        p
    }

    #[test]
    fn cup_example_parses() {
        let src = r#"
scene "Coffee Cup"

cup = cylinder(top: 40mm, bottom: 30mm, height: 100mm) - hollow(wall: 3mm)
handle = torus(radius: 20mm, tube: 6mm) rotate (90deg, 0, 0) at (35mm, 25mm, 0)
add cup, handle

material cup: ceramic
color cup: ivory

ask "mass?"
ask "will it float?"
"#;
        let p = parse_ok(src);
        assert_eq!(p.stmts.len(), 8);
    }

    #[test]
    fn pattern_with_magic() {
        let src = "rungs = repeat(n: 21) cylinder(r: 5mm, h: 33cm) rotate (90deg, 0, 0) rotate (0, i * 30deg, 0) at (0, i * 3.5cm, 0) color: limegreen";
        let p = parse_ok(src);
        assert_eq!(p.stmts.len(), 1);
    }

    #[test]
    fn define_with_params() {
        let src = "define gear(r) = group(cylinder(r: r, height: 6mm), ring(n: 20, radius: r + 3mm) tooth rotate (0, a, 0))";
        parse_ok(src);
    }

    #[test]
    fn multiline_list() {
        let src = "pawn = revolve [(0, 0), (13mm, 0), (13mm, 4mm), (9mm, 8mm), (9mm, 28mm),
                       (14mm, 33mm), (6mm, 36mm)] + sphere(r: 9mm) at (0, 36mm, 0)";
        parse_ok(src);
    }

    #[test]
    fn bare_and_loose_args() {
        parse_ok("star = extrude [(0cm,3cm), (1cm,1cm)] depth: 5mm");
        parse_ok("sphere 5cm");
        parse_ok("cube(12cm, 4cm, 13cm)");
    }

    // ---- 2.1 syntax ----

    #[test]
    fn if_block_and_one_liner() {
        parse_ok("if 2 > 1\n  cube 1cm\nend");
        parse_ok("if 2 > 1: cube 1cm");
        parse_ok("if 2 > 1: cube 1cm else: sphere 1cm");
        parse_ok("if x > 5\n  cube 1cm\nelse if x > 2\n  sphere 1cm\nelse\n  cone 1cm\nend");
        parse_ok("if true\n  cube 1cm\nend");
    }

    #[test]
    fn for_range_and_lists() {
        parse_ok("for i = 1 to 10\n  cube 1cm at (i * 2cm, 0, 0)\nend");
        parse_ok("for i = 1 to 10 by 2: cube 1cm");
        parse_ok("for thing in [1, 2, 3]\n  sphere 1cm\nend");
        parse_ok("for i in 1..5: cube 1cm");
        parse_ok("for a = 0deg to 180deg by 30deg\n  cube 1cm rotate (0, a, 0)\nend");
    }

    #[test]
    fn while_and_flow_control() {
        parse_ok("while x < 5cm\n  x += 1cm\nend");
        parse_ok("while x < 5cm: x += 1cm");
        parse_ok("for i in 1..10\n  if i is 5\n    continue\n  end\n  cube 1cm\nend");
        parse_ok("while true\n  break\nend");
    }

    #[test]
    fn compound_assignment_parses() {
        parse_ok("x = 1\nx += 2\nx -= 1\nx *= 10\nx /= 5");
    }

    #[test]
    fn operators_and_precedence() {
        parse_ok("x = 2 ^ 10 % 7");
        parse_ok("ok = 1 < 2 and 3 > 2 or not 5 == 5");
        parse_ok("ok = 5 is not 4 and 3 in [1, 2, 3]");
        parse_ok("x = -2 ^ 2");
        parse_ok("x = 7 mod 3");
    }

    #[test]
    fn indexing_and_slicing() {
        parse_ok("xs = [1, 2, 3]\nx = xs[0]\ntail = xs[1..3]");
        parse_ok("first = xs[0] + 1");
    }

    #[test]
    fn define_block_template() {
        parse_ok("define wheel(r)\n  rim = torus(radius: r, tube: 1cm)\n  rim\nend\nuse wheel(r: 3cm)");
    }

    #[test]
    fn assert_parses() {
        parse_ok("assert 1 < 2");
        parse_ok("assert x > 0, \"x must be positive\"");
    }

    #[test]
    fn missing_end_is_friendly() {
        let p = parse("if 1 > 0\n cube 1cm", "cm");
        assert!(p.errors.iter().any(|e| e.msg.contains("needs its end")), "{:?}", p.errors);
    }

    #[test]
    fn stray_end_is_friendly() {
        let p = parse("end", "cm");
        assert!(p.errors.iter().any(|e| e.msg.contains("end doesn't close anything")));
    }

    #[test]
    fn stray_else_is_friendly() {
        let p = parse("else\n cube 1cm\nend", "cm");
        assert!(p.errors.iter().any(|e| e.msg.contains("else has no if")));
    }

    #[test]
    fn precedence_shapes_still_work() {
        // the classic forms must parse exactly as before
        parse_ok("cup = cylinder(top: 4cm, bottom: 3cm, height: 10cm) - hollow(wall: 3mm)");
        parse_ok("arch = ring(n: 13, radius: 19cm, from: 0deg, to: 180deg, axis: z) cube(12cm, 4cm, 13cm) rotate (0, 0, a + 90deg)");
    }
}
