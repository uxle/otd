//! P0110 — the abstract syntax tree (2.1 syntax expansion included).

use crate::units::Qty;

#[derive(Clone, Debug)]
pub enum Expr {
    Num(Qty),
    Str(String),
    /// identifier: variable reference, template name, magic var, or value word
    Ident(String),
    /// true / false (2.1)
    Bool(bool),
    /// (a, b) or (a, b, c) — profile point or position
    Tuple(Vec<Expr>),
    /// [ ... ] — list of values (profiles, paths)
    List(Vec<Expr>),
    /// + - & * / % ^ and comparisons and logic
    Bin(BinOp, Box<Expr>, Box<Expr>),
    /// unary minus
    Neg(Box<Expr>),
    /// !x / not x (2.1)
    Not(Box<Expr>),
    /// a..b — inclusive range (2.1)
    Range(Box<Expr>, Box<Expr>),
    /// xs[i] or xs[a..b] — list indexing / slicing (2.1)
    Index(Box<Expr>, Box<Expr>),
    /// shape / builder / pattern / function / template call.
    /// `args` holds named + positional args; bare number/list args become
    /// positional args. For patterns, `body` is the swallowed chain.
    Call {
        name: String,
        line: usize,
        args: Vec<Arg>,
        body: Option<Box<Expr>>,
    },
    Postfix {
        base: Box<Expr>,
        mods: Vec<Mod>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    And, // intersect
    Mul,
    Div,
    // ---- 2.1 ----
    Pow,        // ^
    Mod,        // % / mod
    Lt,         // <
    Gt,         // >
    Le,         // <=
    Ge,         // >=
    Eq,         // == / is
    Ne,         // != / is not
    AndAlso,    // && / and
    OrElse,     // || / or
    In,         // membership: x in xs
}

impl BinOp {
    /// Source spelling for error messages.
    pub fn spell(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::And => "&",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Pow => "^",
            BinOp::Mod => "%",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::AndAlso => "&&",
            BinOp::OrElse => "||",
            BinOp::In => "in",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Arg {
    pub name: Option<String>,
    pub val: Expr,
}

#[derive(Clone, Debug)]
pub enum Mod {
    At(Vec<Expr>),        // rest-point placement (tuple)
    Rotate(Expr),         // angle or (x,y,z) tuple
    /// OTD4 — `rotate (...) pivot (px, py, pz)` — override the default
    /// bounding-box center pivot. `pivot: origin` means the world origin
    /// (0, 0, 0); `pivot: center` is the bbox center (the default, kept
    /// for explicitness); a tuple is a literal pivot point.
    RotateWithPivot { angles: Expr, pivot: PivotSpec },
    Scale(Expr),          // factor or (x,y,z) tuple
    Mirror(char),         // x / y / z world plane
    Material(String),
    Color(String),
    /// 2.0 deep tier: DEC cotan-Laplacian relaxation — `smooth(n: 2, strength: 0.5)`
    Smooth { iterations: Expr, strength: Expr },
    /// 2.0 deep tier: Loop subdivision — `subdiv(n: 2)`
    Subdiv(Expr),
}

/// OTD4 — pivot specification for `rotate`.
#[derive(Clone, Debug)]
pub enum PivotSpec {
    /// the world origin (0, 0, 0)
    Origin,
    /// the bounding-box center (the default — kept for explicitness)
    Center,
    /// a literal (x, y, z) pivot point
    Point(Vec<Expr>),
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Scene(String),
    Version(f64),
    Unit(String),
    Gravity(String),       // "off" | "earth" | "moon" | "mars" | numeric as string
    Camera(String),
    Hide(String),
    Show(String),
    /// P1430 — `mix: A + B`: the chemistry verdict for two materials
    /// (liquids: miscible/immiscible + layers; gases: mix or REACT)
    Mix { a: String, b: String, line: usize },
    /// P2110 (OTD3) — `temperature: 800` | `temperature: 350K` | `temperature: 72F`
    /// — the scene-wide temperature that drives heat/sound/phase physics
    Temperature(String),
    /// P2250 (OTD3.3) — `particle: gluon` | `particle: proton` — one
    /// particle's card from the Standard Model deck
    Particle(String),
    /// P1420b — `environment: air | vacuum | water | oil | density <n>`
    /// sets the medium the whole scene lives in (buoyancy + drag)
    Environment(String),
    /// OTD4 P2300 — `include "parts/wheel.otd"`: load an .otd file as a part
    /// library and merge its top-level parts into the current scene. Powers
    /// the multi-part design workflow: make each part in its own file, then
    /// a main file `include`s them all and arranges them with `at`.
    Include { file: String, at: Option<Vec<Expr>>, line: usize },
    /// OTD4 P2310 — `magnetize: name`: mark a part as a permanent magnet
    /// (overrides its material's intrinsic magnetism). The moment is computed
    /// from the part's volume × a neodymium-grade magnetisation unless a
    /// `moment:` parameter is supplied.
    Magnetize { target: String, moment: Option<Expr>, line: usize },
    /// OTD4 P2320 — `strict: overlap` (or `strict: all`): turn silent-wrongness
    /// into loud errors. With `overlap`, any pair of solid parts that
    /// interpenetrate by more than 0.02 mm fails compilation with a corrective
    /// error rather than a warning.
    Strict(String),
    /// OTD4 P2330 — `overlap: check` (or just `overlap`): the explicit overlap
    /// audit. Reports every interpenetrating pair with the penetration depth
    /// and the shallowest escape axis; under `strict: overlap` it is an error.
    Overlap { mode: String, line: usize },
    /// OTD6 #5 — `connect: A B`: declare electrical connectivity between
    /// two named parts. Powers `simulate: circuit` which walks the real
    /// resistance of modeled windings and reports current/voltage drop.
    Connect { a: String, b: String, line: usize },
    Define { name: String, params: Vec<String>, body: Expr, line: usize },
    /// define name(params) <newline> statements… end — a multi-statement
    /// template; the value is the last expression (2.1)
    DefineBlock { name: String, params: Vec<String>, stmts: Vec<Stmt>, line: usize },
    Use { name: String, args: Vec<Arg>, mods: Vec<Mod>, line: usize },
    Assign(String, Expr, usize),
    /// x += expr / -= / *= / /= (2.1)
    AssignOp { name: String, op: BinOp, val: Expr, line: usize },
    Add { first: Expr, rest: Vec<Expr>, line: usize }, // fuse rest into first
    Apply { kind: ApplyKind, target: Option<String>, value: String, line: usize },
    Simulate(String, usize),
    Ask(String, usize),
    Export { fmt: String, file: String, line: usize },
    Print(String, usize),
    /// if cond … else if cond2 … else … end (2.1). Arms in order; the last
    /// arm is the plain `else` (its condition is the constant true).
    If { arms: Vec<(Expr, Vec<Stmt>)>, els: Option<Vec<Stmt>>, line: usize },
    /// for i = 1 to 10 by 2 … end — counts by step, inclusive ends (2.1)
    For { var: String, from: Expr, to: Expr, step: Option<Expr>, body: Vec<Stmt>, line: usize },
    /// for x in list-or-range … end (2.1)
    ForIn { var: String, iter: Expr, body: Vec<Stmt>, line: usize },
    /// while cond … end (2.1)
    While { cond: Expr, body: Vec<Stmt>, line: usize },
    /// leave the innermost for/while (2.1)
    Break(usize),
    /// skip to the next pass of the innermost for/while (2.1)
    Continue(usize),
    /// assert cond, "message" — stops with a friendly error when false (2.1)
    Assert { cond: Expr, msg: Option<String>, line: usize },
    ExprStmt(Expr, usize),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ApplyKind {
    Material,
    Color,
}

/// A whole parsed program.
pub struct Program {
    pub stmts: Vec<Stmt>,
    pub errors: Vec<crate::lang::errors::Error>,
}
