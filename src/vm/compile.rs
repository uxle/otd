//! P0660 — the expression → bytecode compiler. Walks a pure-numeric
//! `Expr` (numbers with units, numeric variables incl. the pattern magic
//! `i`/`j`/`a`, `pi`, arithmetic, the 8 math functions) and emits OTD-ASM.
//! Anything non-numeric (strings, shapes, lists, shape calls) returns
//! `NotNumeric` and the evaluator falls back to the 1.0 tree-walk — the
//! VM never changes semantics, only the execution engine.

use crate::lang::ast::{Arg, BinOp, Expr};
use crate::units::{Dim, Qty};
use crate::vm::isa::*;

pub const MAX_REGS: u32 = 200;

/// A compiled program: 32-byte instructions + the variable table.
pub struct Program {
    pub insns: Vec<Insn>,
    /// variable names, OP_VAR_F64's `a` indexes this
    pub vars: Vec<String>,
    /// scene default unit × this = mm (for Plain↔Length promotion)
    pub unit_mm: f64,
    /// the register holding the result
    pub result_reg: u32,
}

/// Why compilation bailed — NotNumeric is normal (shapes etc.), TooDeep
/// means the register file would overflow (fallback is still correct).
#[derive(Debug)]
pub enum CompileErr {
    NotNumeric,
    TooDeep,
}

struct Ctx {
    insns: Vec<Insn>,
    vars: Vec<String>,
    next_reg: u32,
}

pub fn compile(e: &Expr, unit_mm: f64) -> Result<Program, CompileErr> {
    let mut ctx = Ctx { insns: Vec::new(), vars: Vec::new(), next_reg: 0 };
    let r = compile_expr(&mut ctx, e)?;
    ctx.insns.push(Insn::new(OP_HALT, 0, 0, 0, 0.0));
    Ok(Program { insns: ctx.insns, vars: ctx.vars, unit_mm, result_reg: r })
}

fn compile_expr(c: &mut Ctx, e: &Expr) -> Result<u32, CompileErr> {
    if c.next_reg >= MAX_REGS {
        return Err(CompileErr::TooDeep);
    }
    let dst = c.next_reg;
    c.next_reg += 1;
    match e {
        Expr::Num(q) => {
            c.insns.push(Insn::new(OP_LOAD_F64, dst, 0, 0, q.v));
            let tag = match q.dim {
                Dim::Length => OP_UNIT_MM,
                Dim::Angle => OP_UNIT_DEG,
                Dim::Plain => OP_UNIT_PLAIN,
            };
            c.insns.push(Insn::new(tag, dst, 0, 0, 0.0));
            Ok(dst)
        }
        Expr::Ident(name) => {
            if name == "pi" {
                c.insns.push(Insn::new(OP_CONST_PI, dst, 0, 0, 0.0));
                c.insns.push(Insn::new(OP_UNIT_PLAIN, dst, 0, 0, 0.0));
                return Ok(dst);
            }
            // numeric variable — resolution happens at run time, so magic
            // vars re-resolve every pattern iteration
            let slot = match c.vars.iter().position(|v| v == name) {
                Some(i) => i,
                None => {
                    c.vars.push(name.clone());
                    c.vars.len() - 1
                }
            };
            c.insns.push(Insn::new(OP_VAR_F64, dst, slot as u32, 0, 0.0));
            Ok(dst)
        }
        Expr::Neg(inner) => {
            let r = compile_expr(c, inner)?;
            c.insns.push(Insn::new(OP_NEG_F64, r, 0, 0, 0.0));
            Ok(r)
        }
        Expr::Bin(op, l, r) => {
            let ra = compile_expr(c, l)?;
            let rb = compile_expr(c, r)?;
            let code = match op {
                BinOp::Add => OP_ADD_F64,
                BinOp::Sub => OP_SUB_F64,
                BinOp::Mul => OP_MUL_F64,
                BinOp::Div => OP_DIV_F64,
                // 2.1 operators stay in the tree-walk: `^` and `mod` carry
                // dimension checks, comparisons answer true/false, and
                // and/or short-circuit — none of which belongs in an eager
                // numeric machine. The VM never changes semantics.
                BinOp::Pow | BinOp::Mod | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
                | BinOp::Eq | BinOp::Ne | BinOp::AndAlso | BinOp::OrElse | BinOp::In
                | BinOp::And => return Err(CompileErr::NotNumeric), // shape op / logic
            };
            // result goes into a fresh register (ra/rb stay live for no
            // reason, but register pressure is bounded by MAX_REGS)
            if c.next_reg >= MAX_REGS {
                return Err(CompileErr::TooDeep);
            }
            let dst2 = c.next_reg;
            c.next_reg += 1;
            c.insns.push(Insn::new(code, dst2, ra, rb, 0.0));
            Ok(dst2)
        }
        Expr::Call { name, args, .. } => {
            // math functions only — shape calls are not numeric
            match name.as_str() {
                "cos" | "sin" | "tan" | "sqrt" | "abs" | "round" => {
                    if args.len() != 1 || args[0].name.is_some() {
                        return Err(CompileErr::NotNumeric);
                    }
                    let r = compile_expr(c, &args[0].val)?;
                    let code = match name.as_str() {
                        "cos" => OP_COS_D,
                        "sin" => OP_SIN_D,
                        "tan" => OP_TAN_D,
                        "sqrt" => OP_SQRT_F64,
                        "abs" => OP_ABS_F64,
                        _ => OP_ROUND_F64,
                    };
                    c.insns.push(Insn::new(code, r, 0, 0, 0.0));
                    // functions return plain numbers (matches eval_func)
                    c.insns.push(Insn::new(OP_UNIT_PLAIN, r, 0, 0, 0.0));
                    Ok(r)
                }
                "min" | "max" => {
                    let vals: Vec<&Arg> = args.iter().filter(|a| a.name.is_none()).collect();
                    if vals.len() < 2 || vals.len() != args.len() {
                        return Err(CompileErr::NotNumeric);
                    }
                    let code = if name == "min" { OP_MIN_F64 } else { OP_MAX_F64 };
                    let mut acc = compile_expr(c, &vals[0].val)?;
                    for v in &vals[1..] {
                        let r = compile_expr(c, &v.val)?;
                        c.insns.push(Insn::new(code, acc, acc, r, 0.0));
                    }
                    // min/max fold raw values (units ignored, like eval_func)
                    c.insns.push(Insn::new(OP_UNIT_PLAIN, acc, 0, 0, 0.0));
                    Ok(acc)
                }
                _ => Err(CompileErr::NotNumeric),
            }
        }
        _ => Err(CompileErr::NotNumeric),
    }
}

impl Program {
    /// Human-readable disassembly (`--vm-dump`).
    pub fn disassemble(&self) -> String {
        let mut out = String::new();
        for (i, insn) in self.insns.iter().enumerate() {
            let mut line = format!("{:04}  ", i);
            line.push_str(&disasm_at(insn, &self.vars));
            out.push_str(&line);
            out.push('\n');
        }
        out
    }
}

/// convenience: is this expression worth a VM attempt?
pub fn is_numeric_shape(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Num(_) | Expr::Ident(_) | Expr::Neg(_) | Expr::Bin(..) | Expr::Call { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::parser;

    fn expr_of(src: &str) -> Expr {
        let prog = parser::parse(src, "cm");
        prog.stmts
            .iter()
            .find_map(|s| match s {
                crate::lang::ast::Stmt::Assign(_, e, _) => Some(e.clone()),
                _ => None,
            })
            .expect("an assignment")
    }

    #[test]
    fn compiles_simple_arithmetic() {
        let e = expr_of("x = 4cm + 3 * 2cm\n");
        let p = compile(&e, 10.0).unwrap();
        assert!(p.insns.len() >= 9);
        assert!(p.insns.last().unwrap().op == OP_HALT);
        // has a load with 40 (4cm = 40mm)
        assert!(p.insns.iter().any(|i| i.op == OP_LOAD_F64 && i.imm == 40.0));
    }

    #[test]
    fn shape_calls_are_not_numeric() {
        let e = expr_of("x = sphere 4cm\n");
        assert!(matches!(compile(&e, 10.0), Err(CompileErr::NotNumeric)));
        let e = expr_of("x = cup - hollow(wall: 3mm)\n");
        assert!(matches!(compile(&e, 10.0), Err(CompileErr::NotNumeric)));
    }

    #[test]
    fn vars_and_pi() {
        let e = expr_of("x = i * pi\n");
        let p = compile(&e, 10.0).unwrap();
        assert!(p.vars.iter().any(|v| v == "i"));
        assert!(p.insns.iter().any(|i| i.op == OP_CONST_PI));
    }

    #[test]
    fn funcs_compile() {
        let e = expr_of("x = sin(30) + cos(a)\n");
        let p = compile(&e, 10.0).unwrap();
        assert!(p.insns.iter().any(|i| i.op == OP_SIN_D));
        assert!(p.insns.iter().any(|i| i.op == OP_COS_D));
        assert!(p.vars.iter().any(|v| v == "a"));
        let e = expr_of("x = min(1, 2, 3)\n");
        let p = compile(&e, 10.0).unwrap();
        assert!(p.insns.iter().filter(|i| i.op == OP_MIN_F64).count() == 2);
    }

    #[test]
    fn deep_expression_falls_back() {
        // ~150 nested additions exceed the register budget
        let mut src = String::from("x = 1");
        for _ in 0..150 {
            src.push_str(" + 1");
        }
        src.push('\n');
        let e = expr_of(&src);
        assert!(matches!(compile(&e, 10.0), Err(CompileErr::TooDeep)));
    }

    #[test]
    fn disassembly_lists_instructions() {
        let e = expr_of("x = 4cm + 1\n");
        let p = compile(&e, 10.0).unwrap();
        let d = p.disassemble();
        assert!(d.contains("V_LOAD_F64"));
        assert!(d.contains("V_ADD_F64"));
        assert!(d.trim_end().ends_with("HALT"));
    }
}
