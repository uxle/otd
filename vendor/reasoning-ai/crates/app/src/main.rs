//! Phase 126 — Interactive CLI: chat with the verified reasoning engine in
//! plain English (Rust port of `python/apps/cli.py`).
//!
//! Usage:
//!     reasoning-ai                     # interactive REPL
//!     reasoning-ai "your question"     # one-shot question
//!     reasoning-ai --json "question"   # one-shot, JSON output (for APIs)
//!
//! Commands inside the REPL: help / explain / quit.
//!
//! Design note: this CLI is a thin shell over engine::ask(); every answer
//! it prints is either VERIFIED (with the verification method shown) or an
//! honest abstention. It never prints an unverified guess as an answer.

use std::collections::HashSet;
use std::io::{self, IsTerminal, Write};

use serde_json::Value;

use reasoning_engine::ask;

const BANNER: &str = r#"
====================================================================
  Verified Reasoning AI  |  math - physics - chemistry - biology
  ------------------------------------------------------------------
  Type a question in plain English. Every answer is independently
  verified before it is shown; if nothing verifies, the engine says
  so instead of guessing.
  Type 'help' for examples, 'explain' to toggle routing details.
====================================================================
"#;

const HELP_TEXT: &str = r#"EXAMPLES (all verified before answering):
  math       What is 12 * 7 + 5?
             Solve 2x + 3 = 11
             Factor x^2 + 5x + 6
             What is 5 choose 2?
             What is gcd(240, 46)?
             What is the determinant of [[1,2],[3,4]]?
  physics    A car accelerates from rest at 3 m/s^2 for 5 s. What is its final velocity?
             A ball is dropped from 20 m. What is its impact speed?
             What is the weight of a 5 kg object?
             How much kinetic energy does a 2 kg object moving at 3 m/s have?
             A 2 kg cart moving at 3 m/s collides with a stationary 3 kg cart...
             What is the current through a 12 V circuit with 4 ohm resistance?
             Three resistors 4 ohm, 6 ohm and 12 ohm are in parallel...
             What is the density of a 5 kg object with volume 2 m^3?
  chemistry  Balance the equation H2 + O2 -> H2O
             What is the molar mass of C6H12O6?
             In N2 + H2 -> NH3, if 2 moles of N2 react with 3 moles of H2, how many moles of NH3 form?
             What is the pH of a solution with [H+] = 1e-3?
             What is the volume of 1 mole of gas at 1 atm and 273 K?
  biology    Cross Aa x Aa. What is the phenotype ratio?
             Cross AaBb x AaBb. What is the phenotype ratio?
             In a Hardy-Weinberg population, 16% shows the recessive trait...
             Transcribe the DNA sequence ATGCAT
             Translate the sequence ATGGCCTAA
             A population of 100 bacteria grows at 10% per hour for 5 hours...
  puzzles    What day of the week was July 4, 1776?
             What is the angle between the hands of a clock at 3:30?
             Alice is the father of Bob. Bob is the father of Carol. How is Alice related to Carol?
             A man walks north 3 km then east 4 km. How far is he from the start?
"#;

fn _color(s: &str, code: &str, enabled: bool) -> String {
    if enabled {
        format!("\x1b[{}m{}\x1b[0m", code, s)
    } else {
        s.to_string()
    }
}

/// Python `f"{x:.0%}"` — percent with 0 decimals.
fn pct(x: f64) -> String {
    format!("{:.0}%", x * 100.0)
}

fn answer_once(question: &str, show_explain: bool, color: bool) {
    let result = ask(question, None, 0);
    let verified = result.get("verified").and_then(Value::as_bool).unwrap_or(false);
    if verified {
        let head = _color("VERIFIED", "1;32", color);
        let answer = result.get("answer").and_then(Value::as_str).unwrap_or("");
        let confidence = result.get("confidence").and_then(Value::as_f64).unwrap_or(0.0);
        println!("  {}: {}   [confidence {}]", head, answer, pct(confidence));
    } else {
        let abstain = _color("ABSTAINED", "1;33", color);
        let why = result
            .get("explanation")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("no verified solution");
        println!("  {}: {}", abstain, why);
    }
    if show_explain {
        let info = result.get("parsed");
        let explain = info
            .and_then(|i| i.get("explain"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let detail = _color(&format!("[route] {}", explain), "2", color);
        println!("  {}", detail);
    }
    println!();
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = argv.iter().filter(|a| !a.starts_with("--")).map(String::as_str).collect();
    let flags: HashSet<String> = argv
        .iter()
        .filter(|a| a.starts_with("--"))
        .map(|a| a.split('=').next().unwrap_or(a).to_string())
        .collect();
    let color = io::stdout().is_terminal() && !flags.contains("--no-color");

    if flags.contains("--json") {
        let question = args.join(" ");
        match serde_json::to_string_pretty(&ask(&question, None, 0)) {
            Ok(s) => println!("{}", s),
            Err(e) => eprintln!("json serialization failed: {}", e),
        }
        return;
    }

    if !args.is_empty() {
        answer_once(&args.join(" "), flags.contains("--explain"), color);
        return;
    }

    println!("{}", BANNER);
    let mut show_explain = false;
    loop {
        print!("{}", _color("you> ", "1;36", color));
        let _ = io::stdout().flush();
        let mut line = String::new();
        // Python caught EOFError and KeyboardInterrupt; std Rust reads can
        // only observe EOF (Ctrl-C terminates the process via SIGINT).
        match io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => {
                println!("\nbye.");
                return;
            }
            Ok(_) => {}
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let low = line.to_lowercase();
        if matches!(low.as_str(), "quit" | "exit" | "q" | "bye") {
            println!("bye.");
            return;
        }
        if matches!(low.as_str(), "help" | "?" | "examples") {
            println!("{}", HELP_TEXT);
            continue;
        }
        if low == "explain" {
            show_explain = !show_explain;
            println!(
                "  routing details {}\n",
                if show_explain { "ON" } else { "OFF" }
            );
            continue;
        }
        answer_once(&line, show_explain, color);
    }
}
