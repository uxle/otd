//! Expression tree + sympy-style Display.

use reasoning_common::Rat;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FuncKind {
    Sin,
    Cos,
    Tan,
    Log,
    Exp,
    Abs,
    Sqrt,
}

impl FuncKind {
    pub fn name(self) -> &'static str {
        match self {
            FuncKind::Sin => "sin",
            FuncKind::Cos => "cos",
            FuncKind::Tan => "tan",
            FuncKind::Log => "log",
            FuncKind::Exp => "exp",
            FuncKind::Abs => "abs",
            FuncKind::Sqrt => "sqrt",
        }
    }

    pub fn from_name(s: &str) -> Option<FuncKind> {
        Some(match s {
            "sin" => FuncKind::Sin,
            "cos" => FuncKind::Cos,
            "tan" => FuncKind::Tan,
            "log" | "ln" => FuncKind::Log,
            "exp" => FuncKind::Exp,
            "abs" => FuncKind::Abs,
            "sqrt" => FuncKind::Sqrt,
            _ => return None,
        })
    }
}

/// Symbolic expression. `sqrt(x)` is represented as `Pow(x, 1/2)`;
/// `pi` and `E` are their own nodes. All numbers are exact rationals.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    /// Exact rational number (integers have den == 1).
    Num(Rat),
    /// Symbol such as `x`.
    Sym(String),
    /// The constant pi.
    Pi,
    /// The constant e (Euler's number).
    E,
    /// Sum — kept canonical by `norm`: flattened, like terms collected,
    /// zero terms dropped, children in canonical sort order.
    Add(Vec<Expr>),
    /// Product — kept canonical by `norm`: numeric coefficient folded out
    /// (represented as a `Num` child first), like bases merged.
    Mul(Vec<Expr>),
    /// Power.
    Pow(Box<Expr>, Box<Expr>),
    /// Function application (sin/cos/tan/log/exp/abs).
    Func(FuncKind, Box<Expr>),
}

impl Expr {
    pub fn int(v: i64) -> Expr {
        Expr::Num(Rat::from_int(v))
    }

    pub fn rat(num: i64, den: i64) -> Expr {
        Expr::Num(Rat::new(num, den))
    }

    pub fn num(v: Rat) -> Expr {
        Expr::Num(v)
    }

    pub fn sym(name: &str) -> Expr {
        Expr::Sym(name.to_string())
    }

    pub fn sqrt(x: Expr) -> Expr {
        Expr::Pow(Box::new(x), Box::new(Expr::rat(1, 2)))
    }

    pub fn is_num(&self) -> bool {
        matches!(self, Expr::Num(_))
    }

    pub fn as_rat(&self) -> Option<Rat> {
        match self {
            Expr::Num(r) => Some(*r),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Expr::Num(r) if r.den == 1 => Some(r.num),
            _ => None,
        }
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, Expr::Num(r) if r.num == 0)
    }

    pub fn is_one(&self) -> bool {
        matches!(self, Expr::Num(r) if r.num == 1 && r.den == 1)
    }

    /// All free symbol names, in first-appearance order.
    pub fn free_symbols(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        self.collect_symbols(&mut out);
        out
    }

    fn collect_symbols(&self, out: &mut Vec<String>) {
        match self {
            Expr::Sym(s) => {
                if !out.contains(s) {
                    out.push(s.clone());
                }
            }
            Expr::Num(_) | Expr::Pi | Expr::E => {}
            Expr::Add(xs) | Expr::Mul(xs) => {
                for x in xs {
                    x.collect_symbols(out);
                }
            }
            Expr::Pow(_, _) => {
                let (a, b) = match self {
                    Expr::Pow(a, b) => (a, b),
                    _ => unreachable!(),
                };
                a.collect_symbols(out);
                b.collect_symbols(out);
            }
            Expr::Func(_, a) => a.collect_symbols(out),
        }
    }

    /// Does this expression contain the given symbol?
    pub fn has_sym(&self, name: &str) -> bool {
        match self {
            Expr::Sym(s) => s == name,
            Expr::Num(_) | Expr::Pi | Expr::E => false,
            Expr::Add(xs) | Expr::Mul(xs) => xs.iter().any(|x| x.has_sym(name)),
            Expr::Pow(a, b) => a.has_sym(name) || b.has_sym(name),
            Expr::Func(_, a) => a.has_sym(name),
        }
    }

    /// Substitute a numeric value for a symbol (returns the substituted
    /// expression, un-normalized — call `simplify`/`norm` to fold).
    pub fn subs(&self, name: &str, value: Expr) -> Expr {
        match self {
            Expr::Sym(s) if s == name => value,
            Expr::Sym(_) | Expr::Num(_) | Expr::Pi | Expr::E => self.clone(),
            Expr::Add(xs) => Expr::Add(xs.iter().map(|x| x.subs(name, value.clone())).collect()),
            Expr::Mul(xs) => Expr::Mul(xs.iter().map(|x| x.subs(name, value.clone())).collect()),
            Expr::Pow(a, b) => Expr::Pow(
                Box::new(a.subs(name, value.clone())),
                Box::new(b.subs(name, value)),
            ),
            Expr::Func(k, a) => Expr::Func(*k, Box::new(a.subs(name, value))),
        }
    }

    /// Iterate over the terms of a sum (like sympy's `Add.make_args`):
    /// a non-Add expression yields itself.
    pub fn terms(&self) -> Vec<Expr> {
        match self {
            Expr::Add(xs) => xs.clone(),
            other => vec![other.clone()],
        }
    }

    /// `(coefficient, rest)` decomposition — like sympy's `as_coeff_Mul()`.
    /// For `Num(r)` returns `(r, 1)`; for a product with leading numeric
    /// factor returns that factor and the remaining factors; else `(1, self)`.
    pub fn as_coeff_mul(&self) -> (Rat, Expr) {
        match self {
            Expr::Num(r) => (*r, Expr::int(1)),
            Expr::Mul(xs) => {
                let mut rest: Vec<Expr> = Vec::new();
                let mut coeff = Rat::from_int(1);
                for (i, x) in xs.iter().enumerate() {
                    if i == 0 {
                        if let Expr::Num(r) = x {
                            coeff = *r;
                            continue;
                        }
                    }
                    rest.push(x.clone());
                }
                if rest.is_empty() {
                    (coeff, Expr::int(1))
                } else if rest.len() == 1 {
                    (coeff, rest.pop().unwrap())
                } else {
                    (coeff, Expr::Mul(rest))
                }
            }
            other => (Rat::from_int(1), other.clone()),
        }
    }

    /// Iterate over the factors of a product (like sympy's
    /// `Mul.make_args`): a non-Mul expression yields itself.
    pub fn factors(&self) -> Vec<Expr> {
        match self {
            Expr::Mul(xs) => xs.clone(),
            other => vec![other.clone()],
        }
    }
}

impl fmt::Display for Expr {
    /// sympy-flavored output: `2*x + 3`, `x**2`, `sqrt(3)`, `sin(x)`,
    /// `3/4`, `pi/6`, `-1/2`. Always re-parseable by this crate's parser.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Num(r) => {
                if r.den == 1 {
                    write!(f, "{}", r.num)
                } else {
                    write!(f, "{}/{}", r.num, r.den)
                }
            }
            Expr::Sym(s) => write!(f, "{}", s),
            Expr::Pi => write!(f, "pi"),
            Expr::E => write!(f, "E"),
            Expr::Add(xs) => {
                let mut first = true;
                for x in xs {
                    // negative numeric coefficient: leading "-" or "- " separator
                    let (neg, mag) = term_sign(x);
                    if first {
                        first = false;
                        if neg {
                            write!(f, "-")?;
                        }
                        write!(f, "{}", mag)?;
                    } else {
                        if neg {
                            write!(f, " - ")?;
                        } else {
                            write!(f, " + ")?;
                        }
                        write!(f, "{}", mag)?;
                    }
                }
                Ok(())
            }
            Expr::Mul(xs) => {
                // canonical form keeps a single leading numeric coefficient
                let mut parts: Vec<String> = Vec::new();
                let mut coeff = Rat::from_int(1);
                let mut coeff_seen = false;
                for x in xs {
                    if let Expr::Num(r) = x {
                        coeff = if coeff_seen { coeff * *r } else { *r };
                        coeff_seen = true;
                    } else {
                        parts.push(mul_part_str(x));
                    }
                }
                if !coeff_seen && !parts.is_empty() {
                    // no numeric coefficient
                    return write!(f, "{}", parts.join("*"));
                }
                let mut out = String::new();
                let negative = coeff.num < 0;
                let mag = Rat {
                    num: coeff.num.abs(),
                    den: coeff.den,
                };
                if negative {
                    out.push('-');
                }
                if !(mag.num == 1 && mag.den == 1) || parts.is_empty() {
                    if mag.den == 1 {
                        out.push_str(&mag.num.to_string());
                    } else {
                        out.push_str(&format!("{}/{}", mag.num, mag.den));
                    }
                }
                for p in &parts {
                    if !out.is_empty() && !out.ends_with('-') {
                        out.push('*');
                    } else if !out.is_empty() {
                        // directly after a leading '-'
                        out.push_str(&p);
                        continue;
                    }
                    out.push_str(p);
                }
                write!(f, "{}", out)
            }
            Expr::Pow(base, exp) => {
                // sqrt display
                if let Expr::Num(r) = exp.as_ref() {
                    if *r == Rat::new(1, 2) {
                        return write!(f, "sqrt({})", base);
                    }
                }
                let b = pow_operand_str(base);
                let e = pow_operand_str(exp);
                write!(f, "{}**{}", b, e)
            }
            Expr::Func(k, a) => write!(f, "{}({})", k.name(), a),
        }
    }
}

/// (is_negative, magnitude_display) for a top-level Add term.
fn term_sign(x: &Expr) -> (bool, Expr) {
    match x {
        Expr::Num(r) if r.num < 0 => (
            true,
            Expr::Num(Rat {
                num: -r.num,
                den: r.den,
            }),
        ),
        Expr::Mul(xs) => {
            if let Some(Expr::Num(r)) = xs.first() {
                if r.num < 0 {
                    let mut rest: Vec<Expr> = Vec::with_capacity(xs.len());
                    rest.push(Expr::Num(Rat {
                        num: -r.num,
                        den: r.den,
                    }));
                    rest.extend(xs[1..].iter().cloned());
                    return (true, Expr::Mul(rest));
                }
            }
            (false, x.clone())
        }
        _ => (false, x.clone()),
    }
}

/// An operand inside a product: parenthesize sums and fractions.
fn mul_part_str(x: &Expr) -> String {
    match x {
        Expr::Add(_) => format!("({})", x),
        Expr::Num(r) if r.den != 1 => format!("({})", x),
        _ => x.to_string(),
    }
}

/// An operand inside a power: parenthesize sums, products, fractions, and
/// negative exponents.
fn pow_operand_str(x: &Expr) -> String {
    match x {
        Expr::Num(r) if r.num < 0 || r.den != 1 => format!("({})", x),
        Expr::Add(_) | Expr::Mul(_) => format!("({})", x),
        _ => x.to_string(),
    }
}

// ---- arithmetic operators (normalized) ----

impl std::ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        crate::norm::norm(&Expr::Add(vec![self, rhs]))
    }
}

impl std::ops::Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        crate::norm::norm(&Expr::Add(vec![
            self,
            Expr::Mul(vec![Expr::int(-1), rhs]),
        ]))
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        crate::norm::norm(&Expr::Mul(vec![self, rhs]))
    }
}

impl std::ops::Div for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        crate::norm::norm(&Expr::Mul(vec![
            self,
            Expr::Pow(Box::new(rhs), Box::new(Expr::int(-1))),
        ]))
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        crate::norm::norm(&Expr::Mul(vec![Expr::int(-1), self]))
    }
}
