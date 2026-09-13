//! Phase 018 — Training loop for the Neural PRM (Rust port of
//! `python/neural/train_prm.py`).
//!
//! The Python original bridged the pure-Python data pipeline (Phases
//! 003/004/011: MCTS -> verifier-grounded (state, Q-value) pairs) into a
//! PyTorch loop with a train/val split and early stopping — but "CANNOT BE
//! RUN OR VERIFIED IN THIS SANDBOX (no torch)". This port keeps the exact
//! same loop structure (Adam + BCELoss + per-epoch shuffling + early
//! stopping on val loss with patience 5 / improvement threshold 1e-4) over
//! the from-scratch `NeuralPrm` of Phase 017, so the loop finally runs and
//! the tests actually watch the loss go down.

use crate::neural_prm::NeuralPrm;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use reasoning_tokenizer::MathTokenizer;

/// Python `build_tensor_dataset`'s `(ids_tensor, mask, target_tensor)`
/// triple: all rows are padded to exactly `max_len` (Python
/// `tokenizer.pad(tokenizer.encode(text, add_special=False), max_len)`),
/// so the boolean mask is fully determined by `mask_pad_id`.
#[derive(Debug, Clone)]
pub struct TensorDataset {
    /// Padded token-id rows, each exactly `max_len` long.
    pub ids: Vec<Vec<i64>>,
    /// The token id that marks padding (Python `mask = ids == pad_id`).
    pub mask_pad_id: i64,
    /// Q-value targets in [0, 1].
    pub targets: Vec<f64>,
}

/// Python `build_tensor_dataset(examples, tokenizer, max_len=32)`.
/// `examples`: list of (state_text, target_q_value) pairs, e.g. produced
/// by walking an MCTS tree and calling `state_to_text_*` on each node.
pub fn build_tensor_dataset(
    examples: &[(String, f64)],
    tokenizer: &MathTokenizer,
    max_len: usize,
) -> TensorDataset {
    let ids = examples
        .iter()
        .map(|(text, _)| {
            tokenizer
                .pad(tokenizer.encode(text, false), max_len)
                .into_iter()
                .map(|i| i as i64)
                .collect::<Vec<i64>>()
        })
        .collect();
    let targets = examples.iter().map(|(_, t)| *t).collect();
    TensorDataset {
        ids,
        mask_pad_id: tokenizer.pad_id as i64,
        targets,
    }
}

/// Adam optimizer over a flat parameter vector (Python:
/// `torch.optim.Adam(model.parameters(), lr=lr)`), with bias-corrected
/// first/second moments and torch's default betas (0.9, 0.999) and eps
/// 1e-8. The moments are sized on the first `step` call (the Python
/// optimizer sized them at construction from the module's parameter list;
/// the flat vector carries the same information).
#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    step_count: usize,
}

impl Adam {
    /// Python `torch.optim.Adam(model.parameters(), lr=lr)`.
    pub fn new(lr: f64) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
            step_count: 0,
        }
    }

    /// One update over ALL parameters:
    /// `p -= lr * m_hat / (sqrt(v_hat) + eps)` with bias-corrected moments.
    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        if self.m.len() != params.len() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
            self.step_count = 0;
        }
        self.step_count += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step_count as i32);
        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

/// Python's `history` dict `{"train_loss": [...], "val_loss": [...]}`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainHistory {
    pub train_loss: Vec<f64>,
    pub val_loss: Vec<f64>,
}

/// BCE loss (Python `nn.BCELoss()`, mean reduction); log arguments clamped
/// away from 0/1 like torch's -100 log clamp.
fn bce_loss(preds: &[f64], targets: &[f64]) -> f64 {
    if preds.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f64;
    for (p, t) in preds.iter().zip(targets.iter()) {
        let pc = p.clamp(1e-12, 1.0 - 1e-12);
        total -= t * pc.ln() + (1.0 - t) * (1.0 - pc).ln();
    }
    total / preds.len() as f64
}

/// Python `train_neural_prm(model, train_examples, val_examples,
/// tokenizer, epochs=30, lr=1e-3, batch_size=32, max_len=32, patience=5,
/// seed=0)` — Adam + BCE loss + early stopping on val loss: an epoch only
/// counts as improved if `val_loss < best_val_loss - 1e-4`, and training
/// stops after `patience` consecutive non-improving epochs. The train set
/// is shuffled every epoch with `StdRng::seed_from_u64(seed)` (Python:
/// `torch.randperm` under `torch.manual_seed(seed)`).
///
/// Returns the trained model and the per-epoch loss history.
#[allow(clippy::too_many_arguments)]
pub fn train_neural_prm(
    mut model: NeuralPrm,
    train_examples: &[(String, f64)],
    val_examples: &[(String, f64)],
    tokenizer: &MathTokenizer,
    epochs: usize,
    lr: f64,
    batch_size: usize,
    max_len: usize,
    patience: usize,
    seed: u64,
) -> (NeuralPrm, TrainHistory) {
    let train = build_tensor_dataset(train_examples, tokenizer, max_len);
    let val = build_tensor_dataset(val_examples, tokenizer, max_len);

    let mut optimizer = Adam::new(lr);
    let mut rng = StdRng::seed_from_u64(seed);

    let mut history = TrainHistory::default();
    let mut best_val_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0usize;

    let n = train.ids.len();
    let batch = batch_size.max(1);
    for _epoch in 0..epochs {
        // shuffle the example order for this epoch (Python torch.randperm)
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);

        let mut total_loss = 0.0f64;
        for chunk in perm.chunks(batch) {
            let ids: Vec<Vec<i64>> = chunk.iter().map(|&i| train.ids[i].clone()).collect();
            let targets: Vec<f64> = chunk.iter().map(|&i| train.targets[i]).collect();
            // forward + manual backward + Adam step (Python:
            // optimizer.zero_grad(); loss.backward(); optimizer.step())
            let (loss, grads) = model.loss_and_grads(&ids, &targets, train.mask_pad_id);
            optimizer.step(model.params_mut(), &grads);
            total_loss += loss * chunk.len() as f64;
        }
        let train_loss = total_loss / n.max(1) as f64;

        // validation pass (Python: model.eval() + torch.no_grad())
        let val_preds = model.forward(&val.ids, val.mask_pad_id);
        let val_loss = bce_loss(&val_preds, &val.targets);

        history.train_loss.push(train_loss);
        history.val_loss.push(val_loss);

        if val_loss < best_val_loss - 1e-4 {
            best_val_loss = val_loss;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement += 1;
            if epochs_without_improvement >= patience {
                break;
            }
        }
    }

    (model, history)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ~40 synthetic (text, target) pairs over the MathTokenizer vocab:
    /// target = 1 iff the text contains the digit 5 — a single-token
    /// pattern the tiny transformer must learn to detect, balanced 20/20
    /// and interleaved so both sides of the 32/8 split see both classes.
    /// This is OUR verification that the manual backprop actually works
    /// (the Python phase had none — it never ran).
    fn synth_examples() -> Vec<(String, f64)> {
        let target = |text: &str| if text.contains('5') { 1.0 } else { 0.0 };
        let mut pos: Vec<String> = Vec::new();
        for d in 1..=5 {
            pos.push(format!("5+{}", d));
        }
        for d in 6..=9 {
            pos.push(format!("{}+5", d));
        }
        for d in 1..=6 {
            pos.push(format!("{}*5", d));
        }
        for d in [7u8, 8, 9, 3, 4] {
            pos.push(format!("5*{}", d));
        }
        let mut neg: Vec<String> = Vec::new();
        for a in 1..=4 {
            for b in 6..=9 {
                neg.push(format!("{}+{}", a, b));
            }
        }
        for (a, b) in [(2, 6), (3, 7), (4, 8), (2, 8)] {
            neg.push(format!("{}*{}", a, b));
        }
        assert_eq!(pos.len(), 20);
        assert_eq!(neg.len(), 20);
        let mut out: Vec<(String, f64)> = Vec::with_capacity(40);
        for i in 0..20 {
            out.push((pos[i].clone(), target(&pos[i])));
            out.push((neg[i].clone(), target(&neg[i])));
        }
        out
    }

    #[test]
    fn test_train_neural_prm_loss_decreases() {
        let tok = MathTokenizer::new();
        let examples = synth_examples();
        assert_eq!(examples.len(), 40);
        // 32 train / 8 val split
        let (train_ex, val_ex): (Vec<(String, f64)>, Vec<(String, f64)>) = (
            examples[..32].to_vec(),
            examples[32..].to_vec(),
        );

        // tiny model per the task spec: d_model 16, heads 2, layers 1,
        // ffn 32, max_len 12; lr 0.01 with batch 2 keeps Adam out of the
        // dead-ReLU saddle (larger steps kill the head's ReLUs in a
        // 3-epoch budget and the model collapses to its base-rate bias)
        let model = NeuralPrm::with_seed(tok.vocab_size(), 16, 2, 1, 12, 32, 0.1, 42);
        let (model, history) =
            train_neural_prm(model, &train_ex, &val_ex, &tok, 3, 0.01, 2, 12, 5, 0);

        // (1) loss decreases: last train_loss < first train_loss
        let first = history.train_loss[0];
        let last = *history.train_loss.last().unwrap();
        println!(
            "[neural] backprop verification: train_loss first = {:.6}, last = {:.6} (val {:.6} -> {:.6})",
            first,
            last,
            history.val_loss[0],
            *history.val_loss.last().unwrap()
        );
        assert!(
            last < first,
            "train loss must decrease over epochs (first={}, last={})",
            first,
            last
        );

        // (2) no NaNs anywhere: all parameters and recorded losses finite
        assert!(
            model.params().iter().all(|v| v.is_finite()),
            "NaN/inf in model parameters after training"
        );
        assert!(history
            .train_loss
            .iter()
            .chain(history.val_loss.iter())
            .all(|l| l.is_finite()));

        // (3) all outputs strictly inside (0, 1)
        let ds = build_tensor_dataset(&examples, &tok, 12);
        assert_eq!(ds.ids.len(), 40);
        assert!(ds.ids.iter().all(|row| row.len() == 12));
        assert_eq!(ds.mask_pad_id, tok.pad_id as i64);
        let preds = model.forward(&ds.ids, ds.mask_pad_id);
        assert_eq!(preds.len(), 40);
        for p in &preds {
            assert!(
                p.is_finite() && *p > 0.0 && *p < 1.0,
                "prediction must be in (0,1), got {}",
                p
            );
        }

        // the model actually learned something: positive examples (contain
        // '5') score higher than negative ones on average
        let pos: Vec<f64> = preds
            .iter()
            .zip(examples.iter())
            .filter(|(_, (_, t))| *t == 1.0)
            .map(|(p, _)| *p)
            .collect();
        let neg: Vec<f64> = preds
            .iter()
            .zip(examples.iter())
            .filter(|(_, (_, t))| *t == 0.0)
            .map(|(p, _)| *p)
            .collect();
        let pos_mean: f64 = pos.iter().sum::<f64>() / pos.len() as f64;
        let neg_mean: f64 = neg.iter().sum::<f64>() / neg.len() as f64;
        println!(
            "[neural] mean pred on 'contains 5' = {:.4} vs else = {:.4}",
            pos_mean, neg_mean
        );
        assert!(
            pos_mean > neg_mean,
            "trained model should rank '5'-containing texts higher ({} vs {})",
            pos_mean,
            neg_mean
        );
    }

    #[test]
    fn test_build_tensor_dataset_pads_like_python() {
        let tok = MathTokenizer::new();
        let ds = build_tensor_dataset(&[("1+2".to_string(), 0.5)], &tok, 8);
        // vocab: <PAD>=0 <BOS>=1 <EOS>=2 <UNK>=3, digits "0".."9" = 4..13,
        // operators "+"..="." = 14..23 -> "1+2" = [5, 14, 6] + pads
        let expected: Vec<i64> = [5usize, 14usize, 6usize]
            .iter()
            .copied()
            .chain(std::iter::repeat(tok.pad_id))
            .take(8)
            .map(|i| i as i64)
            .collect();
        assert_eq!(ds.ids[0], expected);
        assert_eq!(ds.targets, vec![0.5]);
        // truncation to max_len, like Python's pad()
        let ds2 = build_tensor_dataset(&[("1+2".to_string(), 1.0)], &tok, 2);
        assert_eq!(ds2.ids[0].len(), 2);
    }
}
