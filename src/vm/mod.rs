//! P0650 — the OTD-ASM virtual machine facade. Expressions compile to
//! 32-byte vector instructions and execute on a unit-tagged f64 register
//! machine whose hot kernels are handwritten assembly. This is Tier 3 of
//! the OTDL v5 architecture, made real in pure Rust + asm.

pub mod isa;
pub mod compile;
pub mod interp;

use crate::units::Qty;

/// Compile an expression and run it with the given variable resolver.
/// Returns None when the expression is not pure-numeric (the evaluator
/// falls back to the tree-walk) and Err(message) for unit errors.
pub fn try_run(
    e: &crate::lang::ast::Expr,
    unit_mm: f64,
    lookup: &dyn Fn(&str) -> Option<Qty>,
) -> Option<Result<Qty, String>> {
    
    // quick gate: only forms that can be numeric
    if !compile::is_numeric_shape(e) {
        return None;
    }
    let prog = match compile::compile(e, unit_mm) {
        Ok(p) => p,
        Err(_) => return None, // shapes / too deep → tree-walk
    };
    // pre-check that every variable resolves numerically NOW; a scope miss
    // means the tree-walk should handle it (templates, value words, errors)
    for name in &prog.vars {
        if name == "pi" {
            continue;
        }
        match lookup(name) {
            Some(_) => {}
            None => return None,
        }
    }
    Some(interp::run(&prog, lookup).map(|v| v.to_qty()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity fuzz: random expressions evaluated by the VM must match a
    /// reference tree-walk that reimplements the 1.0 unit semantics.
    #[test]
    fn parity_fuzz_vm_vs_reference() {
        use crate::lang::parser;
        use crate::units::Dim;

        // deterministic PRNG for test reproducibility
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || -> f64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64)
        };

        // scope: a few numeric vars
        let scope = |n: &str| -> Option<Qty> {
            match n {
                "i" => Some(Qty::plain(3.0)),
                "a" => Some(Qty::deg(45.0)),
                "w" => Some(Qty::mm(25.0)),
                "pi" => Some(Qty::plain(std::f64::consts::PI)),
                _ => None,
            }
        };

        // reference tree-walk (the 1.0 semantics, standalone)
        fn ref_eval(e: &crate::lang::ast::Expr, s: &dyn Fn(&str) -> Option<Qty>, um: f64) -> Option<Qty> {
            use crate::lang::ast::{BinOp, Expr};
            match e {
                Expr::Num(q) => Some(*q),
                Expr::Ident(n) => s(n),
                Expr::Neg(x) => ref_eval(x, s, um).map(|q| Qty { v: -q.v, dim: q.dim }),
                Expr::Bin(op, l, r) => {
                    let a = ref_eval(l, s, um)?;
                    let b = ref_eval(r, s, um)?;
                    let (a, b) = if matches!(op, BinOp::Add | BinOp::Sub) {
                        match (a.dim, b.dim) {
                            (Dim::Plain, Dim::Length) => (Qty::mm(a.v * um), b),
                            (Dim::Length, Dim::Plain) => (a, Qty::mm(b.v * um)),
                            (Dim::Plain, Dim::Angle) => (Qty::deg(a.v), b),
                            (Dim::Angle, Dim::Plain) => (a, Qty::deg(b.v)),
                            _ => (a, b),
                        }
                    } else {
                        (a, b)
                    };
                    match op {
                        BinOp::Add => Some(Qty { v: a.v + b.v, dim: combine(a.dim, b.dim)? }),
                        BinOp::Sub => Some(Qty { v: a.v - b.v, dim: combine(a.dim, b.dim)? }),
                        BinOp::Mul => Some(Qty { v: a.v * b.v, dim: a.dim.mul(b.dim).ok()? }),
                        BinOp::Div => {
                            if b.v == 0.0 { return None; } // divide by zero errors (2.1 parity)
                            Some(Qty { v: a.v / b.v, dim: a.dim.div(b.dim).ok()? })
                        }
                        // 2.1 operators never reach the VM (NotNumeric in
                        // compile.rs) — the parity fuzz only generates + - * /
                        BinOp::Pow | BinOp::Mod | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
                        | BinOp::Eq | BinOp::Ne | BinOp::AndAlso | BinOp::OrElse | BinOp::In
                        | BinOp::And => None,
                    }
                }
                _ => None,
            }
        }
        fn combine(a: Dim, b: Dim) -> Option<Dim> {
            match (a, b) {
                (Dim::Plain, x) | (x, Dim::Plain) => Some(x),
                (x, y) if x == y => Some(x),
                _ => None,
            }
        }

        // build random expressions as source text (parser = real lexer path)
        let ops = ["+", "-", "*", "/"];
        let units = ["", "cm", "mm", "deg"];
        let vars = ["i", "a", "w", "pi"];
        for _ in 0..2000 {
            let depth = 1 + (next() * 3.0) as usize;
            let mut src = String::new();
            for d in 0..depth {
                let term = if next() < 0.5 {
                    let n = (next() * 20.0).round();
                    format!("{}{}", n, units[(next() * 4.0) as usize])
                } else {
                    vars[(next() * 4.0) as usize].to_string()
                };
                if d > 0 {
                    src.push_str(&format!(" {} ", ops[(next() * 4.0) as usize]));
                }
                src.push_str(&term);
            }
            let src = format!("x = {}\n", src);
            let prog = parser::parse(&src, "cm");
            let e = match prog.stmts.iter().find_map(|s| match s {
                crate::lang::ast::Stmt::Assign(_, e, _) => Some(e.clone()),
                _ => None,
            }) {
                Some(e) => e,
                None => continue,
            };
            let vm = try_run(&e, 10.0, &scope);
            let reference = ref_eval(&e, &scope, 10.0);
            match (vm, reference) {
                (Some(Ok(qv)), Some(qr)) => {
                    assert_eq!(qv.dim, qr.dim, "dim mismatch on {}", src);
                    let same = if qr.v.is_nan() {
                        qv.v.is_nan()
                    } else {
                        qv.v == qr.v || (qv.v - qr.v).abs() <= 1e-9 * (1.0 + qr.v.abs())
                    };
                    assert!(same, "value mismatch on {}: vm {} vs ref {}", src, qv.v, qr.v);
                }
                (Some(Err(_)), None) | (None, None) => {}
                (None, Some(_)) => panic!("VM fell back but reference computed: {}", src),
                (Some(Ok(_)), None) => panic!("VM computed but reference bailed: {}", src),
                (Some(Err(m)), Some(_)) => {
                    // reference has no unit errors in its Ok path; mismatch
                    panic!("VM errored ({}), reference computed: {}", m, src)
                }
            }
        }
    }
}
