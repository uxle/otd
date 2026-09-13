//! Phase 017 — Neural Process Reward Model (Rust port of
//! `python/neural/neural_prm.py`).
//!
//! The Python original was written against PyTorch and — by its own header —
//! "CANNOT BE RUN OR VERIFIED IN THIS SANDBOX". This port replaces torch
//! with a from-scratch implementation over plain `Vec<f64>` blocks with a
//! hand-written forward AND backward pass, so the phase finally runs and is
//! verified: the unit tests include a full numerical gradient check and a
//! tiny end-to-end training run (see `train_prm.rs`).
//!
//! Architecture (identical shapes to the Python module):
//! token embedding -> learned positional embedding -> post-LN Transformer
//! encoder (`n_layers` layers, `n_heads` heads, `d_model`, FFN
//! `dim_feedforward`) -> masked mean-pool over non-pad positions -> linear
//! head (d_model -> d_model/2 -> 1, ReLU between) -> sigmoid.
//!
//! Same job as Phase 004's LinearPRM (predict P(state leads to a
//! verified-correct answer)), but reading the raw tokenized state text
//! instead of 5 hand-picked features — the fix for Phase 013's diagnosed
//! capacity ceiling.
//!
//! Parameter storage: every trainable parameter lives in ONE flat
//! `Vec<f64>` whose block order is fixed by `ParamLayout` (tok_emb, pos_emb,
//! per-layer blocks in the order documented on `LayerOff`, then the head).
//! Matrices are row-major `(rows x cols)` with `w[row*cols + col]`; the
//! attention projections map a `d_model`-dim input to a `d_model`-dim
//! output, FFN-1 maps `d_model -> dim_feedforward`, FFN-2 maps
//! `dim_feedforward -> d_model`, and the head maps `d_model -> d_model/2 ->
//! 1. Gradients come back in exactly the same flat layout, so the Adam
//! optimizer (`train_prm`) can treat the whole model as one slice.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reasoning_search::linear_equation_domain::EqState;
use reasoning_search::NtState;

/// LayerNorm epsilon (torch `nn.LayerNorm` default).
const LN_EPS: f64 = 1e-5;
/// Added to attention logits at pad KEY positions. exp() underflow makes
/// their softmax weight exactly 0 — the finite stand-in for torch's `-inf`
/// `src_key_padding_mask` fill (a whole-pad query row degrades to a uniform
/// distribution, whose output is then zeroed by the pooling mask, matching
/// Python's `(x * valid).sum(1) / valid.sum(1).clamp(min=1)`).
const MASK_VALUE: f64 = -1e9;

/// Serialize a NumberTargetDomain state (tuple of (value, expr)) to text
/// the tokenizer can read (Python `state_to_text_number_target`).
pub fn state_to_text_number_target(state: &NtState) -> String {
    state
        .iter()
        .map(|(_, expr)| expr.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

/// Serialize a LinearEquationDomain EqState to text (Python
/// `state_to_text_equation`): `"{lhs}={rhs}"` via `Expr`'s Display.
pub fn state_to_text_equation(state: &EqState) -> String {
    format!("{}={}", state.lhs, state.rhs)
}

/// Offsets of one encoder layer's parameter blocks inside the flat
/// `params` vector. Block order: attention in-projections (q, k, v),
/// out-projection, post-attention LayerNorm, FFN (up + down), post-FFN
/// LayerNorm. Every bias/gain block follows its weight block.
#[derive(Debug, Clone, Copy)]
struct LayerOff {
    /// (d_model x d_model) + bias
    wq: usize,
    bq: usize,
    wk: usize,
    bk: usize,
    wv: usize,
    bv: usize,
    wo: usize,
    bo: usize,
    /// LayerNorm gain / bias (d_model each)
    ln1_g: usize,
    ln1_b: usize,
    /// FFN up (d_model x F) + bias, down (F x d_model) + bias
    f1: usize,
    f1_b: usize,
    f2: usize,
    f2_b: usize,
    ln2_g: usize,
    ln2_b: usize,
}

/// Canonical flat parameter layout of the whole model:
/// `tok_emb | pos_emb | layer blocks... | head_w1 | head_b1 | head_w2 | head_b2`.
#[derive(Debug, Clone)]
struct ParamLayout {
    /// token embedding: (vocab_size x d_model)
    tok_emb: usize,
    /// learned positional embedding: (max_len x d_model)
    pos_emb: usize,
    layers: Vec<LayerOff>,
    /// head layer 1: (d_model x d_model/2) + bias
    head_w1: usize,
    head_b1: usize,
    /// head layer 2: (d_model/2) + scalar bias
    head_w2: usize,
    head_b2: usize,
    total: usize,
}

/// Cached activations of one encoder layer (everything the backward pass
/// needs). Shapes: `x/q/k/v/attn/y/y_ln/z/z_ln` are (B, T, d);
/// `p` is (B, H, T, T); `ff` is (B, T, F); `ln*_mu`/`ln*_sigma` are (B, T).
struct LayerCache {
    x: Vec<Vec<Vec<f64>>>,
    q: Vec<Vec<Vec<f64>>>,
    k: Vec<Vec<Vec<f64>>>,
    v: Vec<Vec<Vec<f64>>>,
    p: Vec<Vec<Vec<Vec<f64>>>>,
    attn: Vec<Vec<Vec<f64>>>,
    y: Vec<Vec<Vec<f64>>>,
    y_ln: Vec<Vec<Vec<f64>>>,
    ln1_mu: Vec<Vec<f64>>,
    ln1_sigma: Vec<Vec<f64>>,
    ff: Vec<Vec<Vec<f64>>>,
    z: Vec<Vec<Vec<f64>>>,
    z_ln: Vec<Vec<Vec<f64>>>,
    ln2_mu: Vec<Vec<f64>>,
    ln2_sigma: Vec<Vec<f64>>,
}

/// Everything the backward pass needs from one forward pass.
struct Cache {
    ids: Vec<Vec<i64>>,
    /// valid[b][t]: position exists within the row AND is not PAD.
    valid: Vec<Vec<bool>>,
    /// Per-row valid-token count, clamped to >= 1 (Python `.clamp(min=1)`).
    counts: Vec<usize>,
    seq_len: usize,
    layers: Vec<LayerCache>,
    /// Masked mean-pool output: (B, d).
    pooled: Vec<Vec<f64>>,
    /// Head hidden after ReLU: (B, d/2).
    h1: Vec<Vec<f64>>,
    preds: Vec<f64>,
}

/// Neural PRM — Phase 017's capacity upgrade over the LinearPRM.
#[derive(Debug, Clone)]
pub struct NeuralPrm {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub max_len: usize,
    pub dim_feedforward: usize,
    /// Kept purely for API parity with the Python constructor. torch's
    /// dropout is identity at eval time and the Python training loop never
    /// ran in that sandbox; this port never reads the value (documented
    /// no-op).
    pub dropout: f64,
    /// All trainable parameters, flat, in the canonical `ParamLayout` order
    /// (see the module docs for the exact block shapes).
    params: Vec<f64>,
    layout: ParamLayout,
}

impl NeuralPrm {
    /// Python `NeuralPRM(vocab_size, d_model=64, n_heads=4, n_layers=2,
    /// max_len=64, dim_feedforward=128, dropout=0.1)`.
    ///
    /// Weight init: the Python module relied on torch defaults (kaiming
    /// linears, N(0,1) embeddings) that could never be run or verified
    /// anywhere; this port uses a deterministic StdRng-seeded
    /// Glorot(Xavier)-uniform scheme instead — `±sqrt(6/(fan_in+fan_out))`
    /// per weight block, zero biases, LayerNorm gains at 1 (same order of
    /// magnitude as torch, better conditioned for the from-scratch port).
    /// Use [`NeuralPrm::with_seed`] to control the seed.
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        max_len: usize,
        dim_feedforward: usize,
        dropout: f64,
    ) -> Self {
        Self::with_seed(
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            max_len,
            dim_feedforward,
            dropout,
            0,
        )
    }

    /// `new` with an explicit init seed (Python relied on ambient global
    /// torch RNG state; here it is local and deterministic).
    pub fn with_seed(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        max_len: usize,
        dim_feedforward: usize,
        dropout: f64,
        seed: u64,
    ) -> Self {
        // torch raises on these configuration errors at construction time
        // (MultiheadEncoder head split / empty Embedding); mirror as asserts.
        assert!(vocab_size >= 1, "vocab_size must be >= 1");
        assert!(max_len >= 1, "max_len must be >= 1");
        assert!(n_heads >= 1, "n_heads must be >= 1");
        assert!(
            d_model % n_heads == 0,
            "d_model ({}) must be divisible by n_heads ({})",
            d_model,
            n_heads
        );
        let layout = build_layout(vocab_size, d_model, n_layers, dim_feedforward, max_len);
        let mut params = vec![0.0f64; layout.total];
        let d = d_model;
        let d2 = d / 2;
        let f = dim_feedforward;
        let mut rng = StdRng::seed_from_u64(seed);

        init_uniform(
            &mut params[layout.tok_emb..layout.tok_emb + vocab_size * d],
            &mut rng,
            vocab_size,
            d,
        );
        init_uniform(
            &mut params[layout.pos_emb..layout.pos_emb + max_len * d],
            &mut rng,
            max_len,
            d,
        );
        for lo in &layout.layers {
            init_uniform(&mut params[lo.wq..lo.wq + d * d], &mut rng, d, d);
            init_uniform(&mut params[lo.wk..lo.wk + d * d], &mut rng, d, d);
            init_uniform(&mut params[lo.wv..lo.wv + d * d], &mut rng, d, d);
            init_uniform(&mut params[lo.wo..lo.wo + d * d], &mut rng, d, d);
            init_uniform(&mut params[lo.f1..lo.f1 + d * f], &mut rng, d, f);
            init_uniform(&mut params[lo.f2..lo.f2 + f * d], &mut rng, f, d);
            for j in 0..d {
                params[lo.ln1_g + j] = 1.0;
                params[lo.ln2_g + j] = 1.0;
            }
            // all bias blocks stay 0.0
        }
        init_uniform(
            &mut params[layout.head_w1..layout.head_w1 + d * d2],
            &mut rng,
            d,
            d2,
        );
        init_uniform(
            &mut params[layout.head_w2..layout.head_w2 + d2],
            &mut rng,
            d2,
            1,
        );

        NeuralPrm {
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            max_len,
            dim_feedforward,
            dropout,
            params,
            layout,
        }
    }

    /// Flat parameter vector in the canonical block order — exposed for
    /// the Adam optimizer in `train_prm` and for tests (gradient checks,
    /// NaN scans).
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Mutable flat parameter vector (see `params`).
    pub fn params_mut(&mut self) -> &mut [f64] {
        &mut self.params
    }

    /// Python `forward(token_ids, attention_mask)` — the mask is derived
    /// here from `pad_id` (every Python call site built it as
    /// `ids == tok.pad_id`). `token_ids` is a batch of (padded) sequences;
    /// every token equal to `pad_id`, and every position beyond a row's
    /// own length, is masked out of both attention and pooling. Returns
    /// one P(success) in (0, 1) per row.
    pub fn forward(&self, token_ids: &[Vec<i64>], pad_id: i64) -> Vec<f64> {
        self.forward_with_cache(token_ids, pad_id).0
    }

    /// Convenience wrapper: score a single token sequence (Python RUN
    /// LOCALLY example: `model(ids, mask)` on one row).
    pub fn predict_one(&self, ids: &[i64], pad_id: i64) -> f64 {
        let batch = vec![ids.to_vec()];
        self.forward(&batch, pad_id)[0]
    }

    /// BCE loss (mean over the batch) and the gradient of that loss w.r.t.
    /// every parameter, in the flat layout — the training-step primitive
    /// used by `train_prm::train_neural_prm` (Python: `loss.backward()`
    /// feeding `torch.optim.Adam`).
    pub(crate) fn loss_and_grads(
        &self,
        token_ids: &[Vec<i64>],
        targets: &[f64],
        pad_id: i64,
    ) -> (f64, Vec<f64>) {
        let (preds, cache) = self.forward_with_cache(token_ids, pad_id);
        let b = preds.len().max(1);
        let mut loss = 0.0f64;
        for (pred, target) in preds.iter().zip(targets.iter()) {
            // keep the log arguments away from 0/1 (torch BCELoss clamps
            // its logs at -100; sigmoid output only reaches 0/1 at |logit|
            // > ~745 anyway)
            let pc = pred.clamp(1e-12, 1.0 - 1e-12);
            loss -= target * pc.ln() + (1.0 - target) * (1.0 - pc).ln();
        }
        loss /= b as f64;
        let grads = self.backward(&cache, targets);
        (loss, grads)
    }

    /// Full forward pass, caching activations for `backward`.
    fn forward_with_cache(&self, token_ids: &[Vec<i64>], pad_id: i64) -> (Vec<f64>, Cache) {
        let b = token_ids.len();
        let t = token_ids.iter().map(|r| r.len()).max().unwrap_or(0);
        let d = self.d_model;
        let d2 = d / 2;

        // valid[b][t]: within the row AND not a pad token
        let valid: Vec<Vec<bool>> = token_ids
            .iter()
            .map(|row| {
                let mut v = vec![false; t];
                for (i, &id) in row.iter().enumerate() {
                    v[i] = id != pad_id;
                }
                v
            })
            .collect();
        // Python: valid.sum(dim=1).clamp(min=1)
        let counts: Vec<usize> = valid
            .iter()
            .map(|row| row.iter().filter(|&&x| x).count().max(1))
            .collect();

        // --- token + positional embeddings ---
        let tok = &self.params[self.layout.tok_emb..self.layout.tok_emb + self.vocab_size * d];
        let pos = &self.params[self.layout.pos_emb..self.layout.pos_emb + self.max_len * d];
        let mut x: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0f64; d]; t]; b];
        for (bi, row) in token_ids.iter().enumerate() {
            for (ti, &id) in row.iter().enumerate() {
                // tokenizer ids are always in range; clamp instead of
                // raising torch's IndexError (documented deviation)
                let idu = (id.max(0) as usize).min(self.vocab_size - 1);
                let pu = ti.min(self.max_len - 1);
                for j in 0..d {
                    x[bi][ti][j] = tok[idu * d + j] + pos[pu * d + j];
                }
            }
        }

        // --- encoder stack ---
        let mut layers: Vec<LayerCache> = Vec::with_capacity(self.n_layers);
        for l in 0..self.n_layers {
            let lc = self.layer_forward(l, x, &valid);
            x = lc.z_ln.clone(); // this layer's output feeds the next
            layers.push(lc);
        }

        // --- masked mean-pool over non-pad positions ---
        let mut pooled = vec![vec![0.0f64; d]; b];
        for bi in 0..b {
            for ti in 0..t {
                if !valid[bi][ti] {
                    continue;
                }
                for j in 0..d {
                    pooled[bi][j] += x[bi][ti][j];
                }
            }
            let inv = 1.0 / counts[bi] as f64;
            for j in 0..d {
                pooled[bi][j] *= inv;
            }
        }

        // --- head: Linear(d, d/2) -> ReLU -> Linear(d/2, 1) ---
        let w1 = &self.params[self.layout.head_w1..self.layout.head_w1 + d * d2];
        let b1 = &self.params[self.layout.head_b1..self.layout.head_b1 + d2];
        let w2 = &self.params[self.layout.head_w2..self.layout.head_w2 + d2];
        let b2 = self.params[self.layout.head_b2];
        let mut h1: Vec<Vec<f64>> = Vec::with_capacity(b);
        let mut preds: Vec<f64> = Vec::with_capacity(b);
        for bi in 0..b {
            let pre = affine(w1, b1, &pooled[bi]);
            let mut hh = pre;
            for val in hh.iter_mut() {
                if *val < 0.0 {
                    *val = 0.0;
                }
            }
            let mut logit = b2;
            for c in 0..d2 {
                logit += w2[c] * hh[c];
            }
            h1.push(hh);
            preds.push(sigmoid(logit));
        }

        let cache = Cache {
            ids: token_ids.to_vec(),
            valid,
            counts,
            seq_len: t,
            layers,
            pooled,
            h1,
            preds: preds.clone(),
        };
        (preds, cache)
    }

    /// Forward through encoder layer `l`. Consumes the layer input `x`
    /// (kept in the cache for the backward pass) and returns the cache.
    fn layer_forward(&self, l: usize, x: Vec<Vec<Vec<f64>>>, valid: &[Vec<bool>]) -> LayerCache {
        let b = x.len();
        let t = if b > 0 { x[0].len() } else { 0 };
        let d = self.d_model;
        let h = self.n_heads;
        let dh = d / h;
        let f = self.dim_feedforward;
        let lo = &self.layout.layers[l];
        let p = &self.params;

        let wq = &p[lo.wq..lo.wq + d * d];
        let bq = &p[lo.bq..lo.bq + d];
        let wk = &p[lo.wk..lo.wk + d * d];
        let bk = &p[lo.bk..lo.bk + d];
        let wv = &p[lo.wv..lo.wv + d * d];
        let bv = &p[lo.bv..lo.bv + d];
        let wo = &p[lo.wo..lo.wo + d * d];
        let bo = &p[lo.bo..lo.bo + d];
        let ln1_g = &p[lo.ln1_g..lo.ln1_g + d];
        let ln1_b = &p[lo.ln1_b..lo.ln1_b + d];
        let f1 = &p[lo.f1..lo.f1 + d * f];
        let f1_b = &p[lo.f1_b..lo.f1_b + f];
        let f2 = &p[lo.f2..lo.f2 + f * d];
        let f2_b = &p[lo.f2_b..lo.f2_b + d];
        let ln2_g = &p[lo.ln2_g..lo.ln2_g + d];
        let ln2_b = &p[lo.ln2_b..lo.ln2_b + d];

        // --- Q/K/V projections (torch packs them in one in_proj; separate
        //     blocks here, mathematically identical) ---
        let mut q = vec![vec![vec![0.0f64; d]; t]; b];
        let mut k = vec![vec![vec![0.0f64; d]; t]; b];
        let mut v = vec![vec![vec![0.0f64; d]; t]; b];
        for bi in 0..b {
            for ti in 0..t {
                q[bi][ti] = affine(wq, bq, &x[bi][ti]);
                k[bi][ti] = affine(wk, bk, &x[bi][ti]);
                v[bi][ti] = affine(wv, bv, &x[bi][ti]);
            }
        }

        // --- per-head scaled dot-product attention with key padding mask ---
        let scale = 1.0 / (dh as f64).sqrt();
        let mut probs = vec![vec![vec![vec![0.0f64; t]; t]; h]; b];
        for bi in 0..b {
            for hi in 0..h {
                let off = hi * dh;
                for ti in 0..t {
                    let mut scores = vec![0.0f64; t];
                    let mut mx = f64::NEG_INFINITY;
                    for si in 0..t {
                        let mut dot = 0.0;
                        for c in 0..dh {
                            dot += q[bi][ti][off + c] * k[bi][si][off + c];
                        }
                        scores[si] = dot * scale;
                        if !valid[bi][si] {
                            scores[si] += MASK_VALUE;
                        }
                        if scores[si] > mx {
                            mx = scores[si];
                        }
                    }
                    // softmax over keys (max-subtracted for stability)
                    let mut zsum = 0.0;
                    for si in 0..t {
                        let e = (scores[si] - mx).exp();
                        probs[bi][hi][ti][si] = e;
                        zsum += e;
                    }
                    for si in 0..t {
                        probs[bi][hi][ti][si] /= zsum;
                    }
                }
            }
        }

        // --- combine heads, out-projection, residual ---
        let mut attn = vec![vec![vec![0.0f64; d]; t]; b];
        for bi in 0..b {
            for ti in 0..t {
                for hi in 0..h {
                    let off = hi * dh;
                    for c in 0..dh {
                        let mut acc = 0.0;
                        for si in 0..t {
                            acc += probs[bi][hi][ti][si] * v[bi][si][off + c];
                        }
                        attn[bi][ti][off + c] = acc;
                    }
                }
            }
        }
        let mut y = vec![vec![vec![0.0f64; d]; t]; b];
        for bi in 0..b {
            for ti in 0..t {
                let out = affine(wo, bo, &attn[bi][ti]);
                for j in 0..d {
                    y[bi][ti][j] = x[bi][ti][j] + out[j];
                }
            }
        }

        // --- post-attention LayerNorm (post-LN encoder, torch default) ---
        let mut y_ln = vec![vec![vec![0.0f64; d]; t]; b];
        let mut ln1_mu = vec![vec![0.0f64; t]; b];
        let mut ln1_sigma = vec![vec![0.0f64; t]; b];
        for bi in 0..b {
            for ti in 0..t {
                let mut mu = 0.0;
                let mut sig = 0.0;
                ln_norm(&y[bi][ti], ln1_g, ln1_b, &mut y_ln[bi][ti], &mut mu, &mut sig);
                ln1_mu[bi][ti] = mu;
                ln1_sigma[bi][ti] = sig;
            }
        }

        // --- position-wise FFN: Linear(d, F) -> ReLU -> Linear(F, d) ---
        let mut ff = vec![vec![vec![0.0f64; f]; t]; b];
        for bi in 0..b {
            for ti in 0..t {
                let pre = affine(f1, f1_b, &y_ln[bi][ti]);
                for (o, val) in ff[bi][ti].iter_mut().zip(pre.into_iter()) {
                    *o = if val > 0.0 { val } else { 0.0 };
                }
            }
        }
        let mut z = vec![vec![vec![0.0f64; d]; t]; b];
        for bi in 0..b {
            for ti in 0..t {
                let out = affine(f2, f2_b, &ff[bi][ti]);
                for j in 0..d {
                    z[bi][ti][j] = y_ln[bi][ti][j] + out[j];
                }
            }
        }

        // --- post-FFN LayerNorm ---
        let mut z_ln = vec![vec![vec![0.0f64; d]; t]; b];
        let mut ln2_mu = vec![vec![0.0f64; t]; b];
        let mut ln2_sigma = vec![vec![0.0f64; t]; b];
        for bi in 0..b {
            for ti in 0..t {
                let mut mu = 0.0;
                let mut sig = 0.0;
                ln_norm(&z[bi][ti], ln2_g, ln2_b, &mut z_ln[bi][ti], &mut mu, &mut sig);
                ln2_mu[bi][ti] = mu;
                ln2_sigma[bi][ti] = sig;
            }
        }

        LayerCache {
            x,
            q,
            k,
            v,
            p: probs,
            attn,
            y,
            y_ln,
            ln1_mu,
            ln1_sigma,
            ff,
            z,
            z_ln,
            ln2_mu,
            ln2_sigma,
        }
    }

    /// Manual backward pass for the mean-over-batch BCE loss: gradients
    /// w.r.t. every parameter, flat in the canonical layout. `targets` are
    /// the BCE labels (Python `loss_fn(preds, targets).backward()`).
    fn backward(&self, cache: &Cache, targets: &[f64]) -> Vec<f64> {
        let b = cache.preds.len();
        let t = cache.seq_len;
        let d = self.d_model;
        let d2 = d / 2;
        let mut grads = vec![0.0f64; self.params.len()];
        let p = &self.params;
        let lay = &self.layout;

        // ---- head backward ----
        // logit = w2 . h1 + b2 ; h1 = relu(W1 pooled + b1) ; dL/dlogit = p - y
        let mut dpooled = vec![vec![0.0f64; d]; b];
        for bi in 0..b {
            let dlogit = (cache.preds[bi] - targets[bi]) / b.max(1) as f64;
            grads[lay.head_b2] += dlogit;
            for c in 0..d2 {
                grads[lay.head_w2 + c] += dlogit * cache.h1[bi][c];
                let dh1 = dlogit * p[lay.head_w2 + c];
                if cache.h1[bi][c] > 0.0 {
                    // ReLU passed: gradient reaches W1/b1/pooled
                    grads[lay.head_b1 + c] += dh1;
                    for i in 0..d {
                        grads[lay.head_w1 + i * d2 + c] += dh1 * cache.pooled[bi][i];
                        dpooled[bi][i] += dh1 * p[lay.head_w1 + i * d2 + c];
                    }
                }
            }
        }

        // ---- masked mean-pool backward (only valid positions get 1/count) ----
        let mut dzln: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0f64; d]; t]; b];
        for bi in 0..b {
            let inv = 1.0 / cache.counts[bi] as f64;
            for ti in 0..t {
                if !cache.valid[bi][ti] {
                    continue;
                }
                for j in 0..d {
                    dzln[bi][ti][j] = dpooled[bi][j] * inv;
                }
            }
        }

        // ---- encoder layers, reverse order; each returns d/dx, which is
        //      d/dz_ln of the previous layer ----
        for l in (0..self.n_layers).rev() {
            dzln = self.layer_backward(l, cache, &dzln, &mut grads);
        }

        // ---- embeddings backward (dzln is d/d(token_emb + pos_emb)) ----
        for bi in 0..b {
            for ti in 0..t {
                if !cache.valid[bi][ti] {
                    continue; // pad positions carry no gradient (masked out)
                }
                let idu = (cache.ids[bi][ti].max(0) as usize).min(self.vocab_size - 1);
                let pu = ti.min(self.max_len - 1);
                for j in 0..d {
                    grads[lay.tok_emb + idu * d + j] += dzln[bi][ti][j];
                    grads[lay.pos_emb + pu * d + j] += dzln[bi][ti][j];
                }
            }
        }
        grads
    }

    /// Backward through encoder layer `l`. `dzln` is dL/d(z_ln) (this
    /// layer's output); returns dL/d(x) (this layer's input) and accumulates
    /// parameter gradients into `grads`.
    fn layer_backward(
        &self,
        l: usize,
        cache: &Cache,
        dzln: &[Vec<Vec<f64>>],
        grads: &mut [f64],
    ) -> Vec<Vec<Vec<f64>>> {
        let lc = &cache.layers[l];
        let valid = &cache.valid;
        let b = lc.x.len();
        let t = if b > 0 { lc.x[0].len() } else { 0 };
        let d = self.d_model;
        let h = self.n_heads;
        let dh = d / h;
        let f = self.dim_feedforward;
        let lo = &self.layout.layers[l];
        let p = &self.params;

        let wq = &p[lo.wq..lo.wq + d * d];
        let wk = &p[lo.wk..lo.wk + d * d];
        let wv = &p[lo.wv..lo.wv + d * d];
        let wo = &p[lo.wo..lo.wo + d * d];
        let f1 = &p[lo.f1..lo.f1 + d * f];
        let f2 = &p[lo.f2..lo.f2 + f * d];
        let ln1_g = &p[lo.ln1_g..lo.ln1_g + d];
        let ln2_g = &p[lo.ln2_g..lo.ln2_g + d];

        // 1. post-FFN LayerNorm: z_ln = LN(z) — dz = dLN/dz applied to dzln
        let mut dz = vec![vec![vec![0.0f64; d]; t]; b];
        for bi in 0..b {
            for ti in 0..t {
                ln_back(
                    &lc.z[bi][ti],
                    ln2_g,
                    &dzln[bi][ti],
                    lc.ln2_mu[bi][ti],
                    lc.ln2_sigma[bi][ti],
                    &mut dz[bi][ti],
                    grads,
                    lo.ln2_g,
                    lo.ln2_b,
                );
            }
        }

        // 2. FFN branch: z = y_ln + ff_out, ff_out = relu(y_ln·F1 + b1)·F2 + b2
        //    dz is dL/d(ff_out) and (via the residual) the starting dL/dy_ln.
        let mut dpre = vec![vec![vec![0.0f64; f]; t]; b]; // dL/d(relu input)
        for bi in 0..b {
            for ti in 0..t {
                let dzo = &dz[bi][ti];
                let ffo = &lc.ff[bi][ti];
                for j in 0..d {
                    grads[lo.f2_b + j] += dzo[j];
                }
                for fi in 0..f {
                    // dL/d(ff post-relu) = dz · F2^T, gated by the ReLU
                    let mut acc = 0.0;
                    for j in 0..d {
                        acc += dzo[j] * f2[fi * d + j];
                        grads[lo.f2 + fi * d + j] += ffo[fi] * dzo[j];
                    }
                    dpre[bi][ti][fi] = if ffo[fi] > 0.0 { acc } else { 0.0 };
                }
            }
        }
        // 3. ff_pre = y_ln·F1 + b1: residual + parameter gradients
        let mut dy_ln = dz.clone(); // z = y_ln + ff_out
        for bi in 0..b {
            for ti in 0..t {
                let yl = &lc.y_ln[bi][ti];
                for fi in 0..f {
                    let dp = dpre[bi][ti][fi];
                    if dp == 0.0 {
                        continue;
                    }
                    grads[lo.f1_b + fi] += dp;
                    for i in 0..d {
                        grads[lo.f1 + i * f + fi] += yl[i] * dp;
                        dy_ln[bi][ti][i] += dp * f1[i * f + fi];
                    }
                }
            }
        }

        // 4. post-attention LayerNorm: y_ln = LN(y)
        let mut dy = vec![vec![vec![0.0f64; d]; t]; b];
        for bi in 0..b {
            for ti in 0..t {
                ln_back(
                    &lc.y[bi][ti],
                    ln1_g,
                    &dy_ln[bi][ti],
                    lc.ln1_mu[bi][ti],
                    lc.ln1_sigma[bi][ti],
                    &mut dy[bi][ti],
                    grads,
                    lo.ln1_g,
                    lo.ln1_b,
                );
            }
        }

        // 5. residual y = x + attn_out: dx starts as dy; dattn_out = dy
        let mut dx = dy.clone();

        // 6. out-projection: attn_out = attn·Wo + bo
        let mut dattn = vec![vec![vec![0.0f64; d]; t]; b];
        for bi in 0..b {
            for ti in 0..t {
                let dyo = &dy[bi][ti];
                let at = &lc.attn[bi][ti];
                for j in 0..d {
                    grads[lo.bo + j] += dyo[j];
                }
                for kk in 0..d {
                    let mut acc = 0.0;
                    for j in 0..d {
                        acc += dyo[j] * wo[kk * d + j];
                        grads[lo.wo + kk * d + j] += at[kk] * dyo[j];
                    }
                    dattn[bi][ti][kk] = acc;
                }
            }
        }

        // 7. attention backward, per head:
        //    attn[ti][off+c] = Σ_si P[ti][si] · V[si][off+c]
        //    P = softmax(score), score[ti][si] = (q[ti]·k[si]) / sqrt(dh)
        let mut dq = vec![vec![vec![0.0f64; d]; t]; b];
        let mut dk = vec![vec![vec![0.0f64; d]; t]; b];
        let mut dv = vec![vec![vec![0.0f64; d]; t]; b];
        let scale = 1.0 / (dh as f64).sqrt();
        for bi in 0..b {
            for hi in 0..h {
                let off = hi * dh;
                // dP[ti][si] = dattn[ti][off..] · V[si][off..]
                let mut dpmat = vec![vec![0.0f64; t]; t];
                for ti in 0..t {
                    for si in 0..t {
                        let mut acc = 0.0;
                        for c in 0..dh {
                            acc += dattn[bi][ti][off + c] * lc.v[bi][si][off + c];
                        }
                        dpmat[ti][si] = acc;
                    }
                }
                for ti in 0..t {
                    // softmax row inner product: Σ_k P[ti][k]·dP[ti][k]
                    let mut rowdot = 0.0;
                    for k2 in 0..t {
                        rowdot += lc.p[bi][hi][ti][k2] * dpmat[ti][k2];
                    }
                    for si in 0..t {
                        let pv = lc.p[bi][hi][ti][si];
                        // dV[si][off+c] += P[ti][si] · dattn[ti][off+c]
                        for c in 0..dh {
                            dv[bi][si][off + c] += pv * dattn[bi][ti][off + c];
                        }
                        if !valid[bi][si] {
                            continue; // masked key: no gradient to scores
                        }
                        // dS = P ⊙ (dP - rowdot); fold in the 1/sqrt(dh) scale
                        let ds = pv * (dpmat[ti][si] - rowdot) * scale;
                        if ds == 0.0 {
                            continue;
                        }
                        for c in 0..dh {
                            dq[bi][ti][off + c] += ds * lc.k[bi][si][off + c];
                            dk[bi][si][off + c] += ds * lc.q[bi][ti][off + c];
                        }
                    }
                }
            }
        }

        // 8. Q/K/V projections: Q = x·Wq + bq, etc.
        for bi in 0..b {
            for ti in 0..t {
                let xo = &lc.x[bi][ti];
                for j in 0..d {
                    let djq = dq[bi][ti][j];
                    let dkj = dk[bi][ti][j];
                    let dvj = dv[bi][ti][j];
                    grads[lo.bq + j] += djq;
                    grads[lo.bk + j] += dkj;
                    grads[lo.bv + j] += dvj;
                    for i in 0..d {
                        dx[bi][ti][i] += djq * wq[i * d + j];
                        dx[bi][ti][i] += dkj * wk[i * d + j];
                        dx[bi][ti][i] += dvj * wv[i * d + j];
                        grads[lo.wq + i * d + j] += xo[i] * djq;
                        grads[lo.wk + i * d + j] += xo[i] * dkj;
                        grads[lo.wv + i * d + j] += xo[i] * dvj;
                    }
                }
            }
        }
        dx
    }
}

/// Build the flat parameter layout (see module docs for block shapes).
fn build_layout(
    vocab_size: usize,
    d_model: usize,
    n_layers: usize,
    dim_feedforward: usize,
    max_len: usize,
) -> ParamLayout {
    let d = d_model;
    let d2 = d / 2;
    let f = dim_feedforward;
    // running offset allocator: each call reserves `len` slots
    struct Blk {
        off: usize,
    }
    impl Blk {
        fn next(&mut self, len: usize) -> usize {
            let o = self.off;
            self.off += len;
            o
        }
    }
    let mut blk = Blk { off: 0 };
    let tok_emb = blk.next(vocab_size * d);
    let pos_emb = blk.next(max_len * d);
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        layers.push(LayerOff {
            wq: blk.next(d * d),
            bq: blk.next(d),
            wk: blk.next(d * d),
            bk: blk.next(d),
            wv: blk.next(d * d),
            bv: blk.next(d),
            wo: blk.next(d * d),
            bo: blk.next(d),
            ln1_g: blk.next(d),
            ln1_b: blk.next(d),
            f1: blk.next(d * f),
            f1_b: blk.next(f),
            f2: blk.next(f * d),
            f2_b: blk.next(d),
            ln2_g: blk.next(d),
            ln2_b: blk.next(d),
        });
    }
    let head_w1 = blk.next(d * d2);
    let head_b1 = blk.next(d2);
    let head_w2 = blk.next(d2);
    let head_b2 = blk.next(1);
    ParamLayout {
        tok_emb,
        pos_emb,
        layers,
        head_w1,
        head_b1,
        head_w2,
        head_b2,
        total: blk.off,
    }
}

/// Glorot(Xavier)-uniform init of one parameter block:
/// `U(±sqrt(6/(fan_in + fan_out)))`.
fn init_uniform(block: &mut [f64], rng: &mut StdRng, fan_in: usize, fan_out: usize) {
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    for v in block.iter_mut() {
        *v = (rng.gen::<f64>() * 2.0 - 1.0) * bound;
    }
}

/// out = x·W + b for a row-major W (n x m): `w[i*m + j]` (n = x.len(),
/// m = b.len()).
fn affine(w: &[f64], b: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let m = b.len();
    let mut out = vec![0.0f64; m];
    for j in 0..m {
        let mut acc = b[j];
        for i in 0..n {
            acc += x[i] * w[i * m + j];
        }
        out[j] = acc;
    }
    out
}

/// Row LayerNorm (torch default: biased variance, eps inside the sqrt):
/// `out = (z - mean) / sqrt(var + eps) * g + b`. Reports the mean and the
/// sigma actually used, for the backward pass.
fn ln_norm(z: &[f64], g: &[f64], b: &[f64], out: &mut [f64], mu: &mut f64, sigma: &mut f64) {
    let n = z.len();
    let m = z.iter().sum::<f64>() / n as f64;
    let var = z.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64;
    let s = (var + LN_EPS).sqrt();
    for j in 0..n {
        out[j] = (z[j] - m) / s * g[j] + b[j];
    }
    *mu = m;
    *sigma = s;
}

/// LayerNorm backward for one row. `dout` = dL/d(out); writes dL/dz into
/// `dz` and accumulates the gain/bias gradients into `grads` at
/// `dg_off`/`db_off`. Standard form:
/// `dz_j = dzhat_j/sigma - Σ dzhat/(n·sigma) - (z_j-mu)·Σ(dzhat·dev)/(n·sigma³)`
/// with `dzhat = dout * g` (the dmu/dvar cross-terms cancel exactly because
/// `Σ dev = 0`).
fn ln_back(
    z: &[f64],
    g: &[f64],
    dout: &[f64],
    mu: f64,
    sigma: f64,
    dz: &mut [f64],
    grads: &mut [f64],
    dg_off: usize,
    db_off: usize,
) {
    let n = z.len();
    let mut sum_dzhat = 0.0;
    let mut sum_dzhat_dev = 0.0;
    for j in 0..n {
        let dzhat = dout[j] * g[j];
        sum_dzhat += dzhat;
        sum_dzhat_dev += dzhat * (z[j] - mu);
    }
    let c1 = sum_dzhat / (n as f64 * sigma);
    let c3 = sum_dzhat_dev / (n as f64 * sigma * sigma * sigma);
    for j in 0..n {
        dz[j] = dout[j] * g[j] / sigma - c1 - (z[j] - mu) * c3;
        grads[dg_off + j] += dout[j] * (z[j] - mu) / sigma;
        grads[db_off + j] += dout[j];
    }
}

/// Numerically stable sigmoid.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reasoning_search::{make_initial_state, Domain, LinearEquationDomain,
                           NumberTargetDomain};
    use reasoning_tokenizer::MathTokenizer;

    #[test]
    fn test_forward_predict_one_in_unit_interval() {
        let tok = MathTokenizer::new();
        let model = NeuralPrm::with_seed(tok.vocab_size(), 16, 2, 1, 12, 32, 0.1, 3);
        let ids: Vec<i64> = tok
            .encode("2+2=4", false)
            .into_iter()
            .map(|i| i as i64)
            .collect();
        let p = model.predict_one(&ids, tok.pad_id as i64);
        assert!(
            p.is_finite() && p > 0.0 && p < 1.0,
            "predict_one must return a probability in (0,1), got {}",
            p
        );

        // batch forward: the same text padded to 12 and left unpadded must
        // produce identical predictions (pad masking works)
        let padded: Vec<i64> = tok
            .pad(tok.encode("2+2=4", false), 12)
            .into_iter()
            .map(|i| i as i64)
            .collect();
        let batch = model.forward(&[padded, ids], tok.pad_id as i64);
        assert_eq!(batch.len(), 2);
        for p in &batch {
            assert!(p.is_finite() && *p > 0.0 && *p < 1.0);
        }
        assert!(
            (batch[0] - batch[1]).abs() < 1e-12,
            "padding must not change the prediction: {} vs {}",
            batch[0],
            batch[1]
        );

        // all parameters finite after a forward pass
        assert!(model.params().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_state_to_text_helpers() {
        // number-target state: exprs joined with ","
        let state = make_initial_state(&[4.0, 7.0]);
        assert_eq!(state_to_text_number_target(&state), "4,7");
        let domain = NumberTargetDomain::new(11.0);
        let next = domain.apply(&state, &(0, 1, '+'));
        assert_eq!(state_to_text_number_target(&next), "(4+7)");

        // equation state: "{lhs}={rhs}" via Expr's Display
        let eq_domain = LinearEquationDomain::try_new("3*x=7", 6).unwrap();
        assert_eq!(state_to_text_equation(&eq_domain.initial_state()), "3*x=7");
    }

    /// THE backprop verification the Python version never had: compare the
    /// analytic gradients against central finite differences of the loss.
    #[test]
    fn test_backward_matches_numerical_gradients() {
        let tok = MathTokenizer::new();
        let model = NeuralPrm::with_seed(tok.vocab_size(), 8, 2, 1, 6, 12, 0.1, 11);
        let pad = tok.pad_id as i64;
        let row = |text: &str| -> Vec<i64> {
            tok.pad(tok.encode(text, false), 6)
                .into_iter()
                .map(|i| i as i64)
                .collect()
        };
        let ids = vec![row("3+4"), row("5*1")];
        let targets = vec![0.0, 1.0];

        let (_, grads) = model.loss_and_grads(&ids, &targets, pad);
        assert!(grads.iter().all(|g| g.is_finite()));

        let n_params = model.params().len();
        let eps = 1e-6;
        let step = (n_params / 24).max(1); // sample ~24 indices spread over
        let mut checked = 0;
        let mut worst = 0.0f64;
        for idx in (0..n_params).step_by(step) {
            let mut plus = model.clone();
            plus.params_mut()[idx] += eps;
            let (lp, _) = plus.loss_and_grads(&ids, &targets, pad);
            let mut minus = model.clone();
            minus.params_mut()[idx] -= eps;
            let (lm, _) = minus.loss_and_grads(&ids, &targets, pad);
            let numeric = (lp - lm) / (2.0 * eps);
            let diff = (numeric - grads[idx]).abs();
            worst = worst.max(diff);
            assert!(
                diff < 1e-4 * (1.0 + numeric.abs().max(grads[idx].abs())),
                "gradient mismatch at param {}: analytic={} numeric={}",
                idx,
                grads[idx],
                numeric
            );
            checked += 1;
        }
        assert!(checked >= 20, "expected to check >= 20 params, got {}", checked);
        println!(
            "[neural] numerical gradient check: {}/{} params, worst |analytic-numeric| = {:e}",
            checked, n_params, worst
        );
    }
}

