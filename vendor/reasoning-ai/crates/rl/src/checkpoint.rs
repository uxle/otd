//! Phase 078 — Checkpointing (design doc 25: "Implement checkpointing,
//! resume training, ... reproducibility").
//!
//! Plain JSON, not pickle: policy weights are a 5-float map, training
//! history is plain lists/floats — no reason to accept pickle's
//! arbitrary-code-execution risk for data this simple. Writes are atomic
//! (tmp file + rename), so no half-written checkpoint on crash.

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::policy::{SoftmaxPolicy, FEATURE_KEYS};

#[derive(Debug, Serialize, Deserialize)]
struct CheckpointPayload {
    weights: HashMap<String, f64>,
    #[serde(default)]
    metadata: serde_json::Value,
    #[serde(default)]
    saved_at: f64,
}

/// Writes `{"weights": {...}, "metadata": {...}, "saved_at": <unix ts>}`
/// to `path` (atomically: tmp file + rename on POSIX).
pub fn save_policy_checkpoint(
    policy: &SoftmaxPolicy,
    path: &str,
    metadata: Option<serde_json::Value>,
) -> Result<(), String> {
    let weights: HashMap<String, f64> = FEATURE_KEYS
        .iter()
        .map(|k| (k.to_string(), policy.weights.get(*k).copied().unwrap_or(0.0)))
        .collect();
    let payload = CheckpointPayload {
        weights,
        metadata: metadata.unwrap_or(serde_json::Value::Object(Default::default())),
        saved_at: unix_time(),
    };
    write_json_atomic(&payload, path)
}

/// Load a policy checkpoint written by [`save_policy_checkpoint`].
/// Rejects unknown feature keys (Python ValueError).
pub fn load_policy_checkpoint(path: &str) -> Result<SoftmaxPolicy, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read checkpoint {}: {}", path, e))?;
    let payload: CheckpointPayload = serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse checkpoint {}: {}", path, e))?;
    let unknown: Vec<&String> = payload
        .weights
        .keys()
        .filter(|k| !FEATURE_KEYS.contains(&k.as_str()))
        .collect();
    if !unknown.is_empty() {
        // Python: f"checkpoint has unknown feature keys: {unknown}" (a set repr)
        let joined: Vec<String> = unknown.iter().map(|k| format!("'{}'", k)).collect();
        return Err(format!(
            "checkpoint has unknown feature keys: {{{}}}",
            joined.join(", ")
        ));
    }
    let full_weights: HashMap<String, f64> = FEATURE_KEYS
        .iter()
        .map(|k| {
            (
                k.to_string(),
                payload.weights.get(*k).copied().unwrap_or(0.0),
            )
        })
        .collect();
    Ok(SoftmaxPolicy {
        weights: full_weights,
    })
}

/// Serializes any serializable training history (TrainingHistory,
/// CurriculumTrainingHistory, ...) to JSON. Python accepted dataclass
/// instances or dicts; in Rust anything `serde::Serialize` works.
pub fn save_training_history<T: Serialize>(history: &T, path: &str) -> Result<(), String> {
    write_json_atomic(history, path)
}

/// Loads a training history back as plain JSON (`serde_json::Value`,
/// the Rust analogue of Python's parsed dict).
pub fn load_training_history(path: &str) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read history {}: {}", path, e))?;
    serde_json::from_str(&text).map_err(|e| format!("failed to parse history {}: {}", path, e))
}

fn write_json_atomic<T: Serialize>(payload: &T, path: &str) -> Result<(), String> {
    let text = serde_json::to_string_pretty(payload)
        .map_err(|e| format!("failed to serialize: {}", e))?;
    let tmp_path = format!("{}.tmp", path);
    std::fs::write(&tmp_path, text).map_err(|e| format!("failed to write {}: {}", tmp_path, e))?;
    // Atomic on POSIX: no half-written checkpoint on crash.
    std::fs::rename(&tmp_path, path).map_err(|e| {
        // Python would leave the tmp file behind on failure too; try to
        // clean it up the way an aborted Python run would not, but keep
        // the original error.
        let _ = std::fs::remove_file(Path::new(&tmp_path));
        format!("failed to rename {} -> {}: {}", tmp_path, path, e)
    })
}

fn unix_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
