//! Phase 126b — JSON bridge between the web API and the verified engine
//! (Rust port of `python/apps/api_entry.py`).
//!
//! Reads ONE JSON object from stdin and writes ONE JSON object to stdout:
//!
//!     {"question": "plain english question"}
//!         -> runs the deterministic NL layer (nlp.solver.ask)
//!
//!     {"typed": {"kind": "...", "payload": {...}}}
//!         -> routes an explicitly typed problem through solve() (this is
//!            how the web API's LLM-fallback path submits a structured
//!            extraction WITHOUT it ever being returned unverified —
//!            solve() still gates on verification)
//!
//! Always exits 0 with {"ok": true/false, ...} so the HTTP layer can
//! distinguish engine abstention from engine failure. (The Python version
//! wrapped everything in `except Exception` — the Rust equivalent is a
//! catch_unwind around the dispatch so a panic never leaks to stdout as
//! if it were an answer. The Python script itself is a stdin->stdout
//! JSON bridge, not an HTTP server; the web layer that shells out to it
//! is outside this port's scope — same spirit, identical protocol.)

use std::io::{Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};

use serde_json::{json, Map, Value};

use reasoning_engine::{ask, solve, Problem};

/// Python truthiness for JSON values (None/False/0/""/{} /[] are falsy).
fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn dispatch(request: &Value) -> Value {
    let typed = request.get("typed").filter(|v| is_truthy(v));
    if let Some(typed) = typed.and_then(Value::as_object) {
        let kind = typed
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        // Python: `request["typed"].get("payload") or {}`
        let payload = match typed.get("payload") {
            Some(Value::Object(m)) => m.clone(),
            _ => Map::new(),
        };
        let ans = solve(&Problem::new(kind.clone(), payload.clone()), None, 0);
        return json!({
            "ok": true,
            "source": "typed",
            "kind": kind,
            "payload": payload,
            "verified": ans.verified,
            "answer": ans.final_answer(),
            "confidence": if ans.verified { ans.confidence } else { 0.0 },
            "explanation": ans.explanation,
        });
    }
    // Python: {"ok": True, "source": "nl", **ask(str(request.get("question", "")))}
    let question = request
        .get("question")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut out = Map::new();
    out.insert("ok".to_string(), json!(true));
    out.insert("source".to_string(), json!("nl"));
    if let Value::Object(asked) = ask(&question, None, 0) {
        for (k, v) in asked {
            out.insert(k, v);
        }
    }
    Value::Object(out)
}

fn main() {
    let mut input = String::new();
    if std::io::stdin().read_to_string(&mut input).is_err() {
        // treat an unreadable stream as an empty request
        input = String::new();
    }
    let trimmed = input.trim();
    let request: Value = if trimmed.is_empty() {
        Value::Object(Map::new())
    } else {
        match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let out = json!({"ok": false, "error": format!("bad JSON request: {}", e)});
                println!("{}", out);
                let _ = std::io::stdout().flush();
                return;
            }
        }
    };

    // never leak a panic to the HTTP layer as "an answer"
    let result = catch_unwind(AssertUnwindSafe(|| dispatch(&request)))
        .unwrap_or_else(|_| json!({"ok": false, "error": "engine error: panic"}));
    match serde_json::to_string(&result) {
        Ok(s) => println!("{}", s),
        Err(e) => println!(
            "{}",
            json!({"ok": false, "error": format!("engine error: {}", e)})
        ),
    }
    let _ = std::io::stdout().flush();
}
