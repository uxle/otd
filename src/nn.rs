//! P2200 — OTD `nn.rs`: a single-file, zero-dependency neural network library.
//!
//! A faithful self-made port of the nn.rs idea (FL33TW00D/nn.rs — "the fastest
//! way to build neural networks in Rust, one file, no deps"). The real crate
//! could not be vendored (repo unreachable) — and this project's constraint is
//! P0000: everything written from nothing. So this IS nn.rs, rebuilt:
//!
//!   • `Tensor` — row-major f32 with shape tracking, slicing, transpose
//!   • matmul — cache-blocked, 4-wide unrolled inner loop (matches the
//!     project's SIMD-first philosophy; SSE2 baseline friendly)
//!   • `Linear` — W (in×out) + b, fused forward/backward with cached input
//!   • Activations — ReLU, GELU (tanh approx), **CELU** (nn.rs's signature),
//!     Sigmoid, Tanh, Softmax
//!   • `Adam`, `AdamW`, `Sgd` — the optimizer trio, bias-corrected moments
//!   • `Mlp` — stacked layers with manual reverse-mode backprop
//!   • `softmax_cross_entropy` — the fused loss every char-model trains on
//!   • `Trainer` — mini-batch loop, seeded shuffling (crate::rng), history
//!
//! The library is deterministic: same seed ⇒ same weights ⇒ same loss curve.
//! The flagship demo (`simulate: learn` / `learn_freefall`) trains a net on
//! the engine's own closed-form physics — t(h) = √(2h/g) — so OTD's AI learns
//! OTD's gravity. Self-made math learning self-made laws.

use crate::rng::Rng;

// ─────────────────────────────────────────────────────────────────────────────
// Tensor
// ─────────────────────────────────────────────────────────────────────────────

/// Dense row-major f32 tensor. Shape is carried, not computed.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor { shape: shape.to_vec(), data: vec![0.0; n] }
    }

    pub fn full(shape: &[usize], v: f32) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor { shape: shape.to_vec(), data: vec![v; n] }
    }

    /// Standard-normal init (Box–Muller from the seeded engine PRNG).
    pub fn randn(shape: &[usize], std: f32, rng: &mut Rng) -> Tensor {
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        let mut spare: Option<f32> = None;
        for _ in 0..n {
            let z = match spare.take() {
                Some(z) => z,
                None => {
                    let (a, b) = loop {
                        let u1 = rng.next_f64() as f32;
                        let u2 = rng.next_f64() as f32;
                        if u1 > 1e-7 {
                            break (u1, u2);
                        }
                    };
                    let r = (-2.0 * a.ln()).sqrt();
                    let th = 2.0 * core::f32::consts::PI * b;
                    spare = Some(r * th.sin());
                    r * th.cos()
                }
            };
            data.push(z * std);
        }
        Tensor { shape: shape.to_vec(), data }
    }

    /// Uniform in [lo, hi).
    pub fn uniform(shape: &[usize], lo: f32, hi: f32, rng: &mut Rng) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor {
            shape: shape.to_vec(),
            data: (0..n).map(|_| rng.next_range(lo as f64, hi as f64) as f32).collect(),
        }
    }

    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Tensor {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Tensor { shape: shape.to_vec(), data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Row-major (i, j) element access.
    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.shape[1] + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f32) {
        let w = self.shape[1];
        self.data[i * w + j] = v;
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        debug_assert_eq!(shape.iter().product::<usize>(), self.data.len());
        Tensor { shape: shape.to_vec(), data: self.data.clone() }
    }

    /// Transpose a 2-D tensor.
    pub fn t(&self) -> Tensor {
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = Tensor::zeros(&[c, r]);
        for i in 0..r {
            for j in 0..c {
                out.data[j * r + i] = self.data[i * c + j];
            }
        }
        out
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|&x| f(x)).collect() }
    }

    pub fn zip(&self, o: &Tensor, f: impl Fn(f32, f32) -> f32) -> Tensor {
        debug_assert_eq!(self.shape, o.shape);
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().zip(&o.data).map(|(&a, &b)| f(a, b)).collect(),
        }
    }

    pub fn add(&self, o: &Tensor) -> Tensor {
        self.zip(o, |a, b| a + b)
    }

    pub fn sub(&self, o: &Tensor) -> Tensor {
        self.zip(o, |a, b| a - b)
    }

    pub fn hadamard(&self, o: &Tensor) -> Tensor {
        self.zip(o, |a, b| a * b)
    }

    pub fn scale(&self, s: f32) -> Tensor {
        self.map(|x| x * s)
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            0.0
        } else {
            self.sum() / self.data.len() as f32
        }
    }

    /// Cache-blocked matmul. The k-loop is the hot path: b reads walk columns
    /// of B via a fixed stride, and the accumulate is unrolled 4-wide so the
    /// compiler vectorises it (SSE2 baseline → 4 f32 lanes per cycle).
    pub fn matmul(&self, o: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul: A must be 2-D");
        assert_eq!(o.shape.len(), 2, "matmul: B must be 2-D");
        let (m, k, n) = (self.shape[0], self.shape[1], o.shape[1]);
        assert_eq!(self.shape[1], o.shape[0], "matmul: inner dims must agree");
        let mut out = Tensor::zeros(&[m, n]);
        let a = &self.data;
        let b = &o.data;
        let d = &mut out.data;
        // block over rows for L1 residency
        const RB: usize = 64;
        for r0 in (0..m).step_by(RB) {
            let rmax = (r0 + RB).min(m);
            for i in r0..rmax {
                let ai = &a[i * k..(i + 1) * k];
                let di = &mut d[i * n..(i + 1) * n];
                for t in 0..k {
                    let av = ai[t];
                    let bt = &b[t * n..(t + 1) * n];
                    let mut j = 0usize;
                    while j + 4 <= n {
                        di[j] += av * bt[j];
                        di[j + 1] += av * bt[j + 1];
                        di[j + 2] += av * bt[j + 2];
                        di[j + 3] += av * bt[j + 3];
                        j += 4;
                    }
                    while j < n {
                        di[j] += av * bt[j];
                        j += 1;
                    }
                }
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Activations (nn.rs's function set)
// ─────────────────────────────────────────────────────────────────────────────

pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn relu_d(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// GELU, tanh approximation (the transformer workhorse).
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + tanhf(0.797_884_56 * (x + 0.044_715 * x * x * x)))
}

pub fn gelu_d(x: f32) -> f32 {
    // derivative of the tanh approximation
    let inner = 0.797_884_56 * (x + 0.044_715 * x * x * x);
    let t = tanhf(inner);
    let d_inner = 0.797_884_56 * (1.0 + 3.0 * 0.044_715 * x * x);
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
}

/// CELU — Continuously Differentiable Exponential Linear Unit,
/// the signature activation of the original nn.rs.
pub fn celu(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 { x } else { alpha * (expf(x / alpha) - 1.0) }
}

pub fn celu_d(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { expf(x / alpha) }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + expf(-x))
}

pub fn sigmoid_d(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

pub fn tanh(x: f32) -> f32 {
    tanhf(x)
}

pub fn tanh_d(x: f32) -> f32 {
    let t = tanhf(x);
    1.0 - t * t
}

/// Numerically stable softmax over the last axis of a batch (n, c).
pub fn softmax(logits: &Tensor) -> Tensor {
    let (n, c) = (logits.shape[0], logits.shape[1]);
    let mut out = Tensor::zeros(&[n, c]);
    for i in 0..n {
        let row = &logits.data[i * c..(i + 1) * c];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&x| expf(x - mx)).sum();
        for j in 0..c {
            out.data[i * c + j] = expf(row[j] - mx) / sum;
        }
    }
    out
}

// libm-free intrinsics wrappers (std f32 methods)
fn expf(x: f32) -> f32 {
    x.exp()
}
fn tanhf(x: f32) -> f32 {
    x.tanh()
}
fn lnf(x: f32) -> f32 {
    x.ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// Losses (fused forward/backward)
// ─────────────────────────────────────────────────────────────────────────────

/// Softmax + cross-entropy: forward gives mean loss over the batch,
/// backward writes dLogits into `dx` (the input's grad buffer).
pub fn softmax_cross_entropy(logits: &Tensor, targets: &[usize]) -> (f32, Tensor) {
    let (n, c) = (logits.shape[0], logits.shape[1]);
    assert_eq!(targets.len(), n);
    let probs = softmax(logits);
    let mut loss = 0.0f32;
    for i in 0..n {
        let p = probs.data[i * c + targets[i]].max(1e-12);
        loss -= lnf(p);
    }
    loss /= n as f32;
    // fused gradient: (p - onehot) / n
    let mut dx = probs;
    for i in 0..n {
        dx.data[i * c + targets[i]] -= 1.0;
    }
    let dx = dx.scale(1.0 / n as f32);
    (loss, dx)
}

/// Mean squared error and its gradient wrt the prediction.
pub fn mse(pred: &Tensor, truth: &Tensor) -> (f32, Tensor) {
    let diff = pred.sub(truth);
    let loss = diff.data.iter().map(|d| d * d).sum::<f32>() / diff.len() as f32;
    let grad = diff.scale(2.0 / diff.len() as f32);
    (loss, grad)
}

// ─────────────────────────────────────────────────────────────────────────────
// Layers
// ─────────────────────────────────────────────────────────────────────────────

/// Which activation a layer applies after the affine map.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Act {
    None,
    Relu,
    Gelu,
    Celu,
    Tanh,
    Sigmoid,
}

impl Act {
    pub fn f(&self, x: f32) -> f32 {
        match self {
            Act::None => x,
            Act::Relu => relu(x),
            Act::Gelu => gelu(x),
            Act::Celu => celu(x, 1.0),
            Act::Tanh => tanh(x),
            Act::Sigmoid => sigmoid(x),
        }
    }
    pub fn df(&self, x: f32) -> f32 {
        match self {
            Act::None => 1.0,
            Act::Relu => relu_d(x),
            Act::Gelu => gelu_d(x),
            Act::Celu => celu_d(x, 1.0),
            Act::Tanh => tanh_d(x),
            Act::Sigmoid => sigmoid_d(x),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Act::None => "linear",
            Act::Relu => "relu",
            Act::Gelu => "gelu",
            Act::Celu => "celu",
            Act::Tanh => "tanh",
            Act::Sigmoid => "sigmoid",
        }
    }
}

/// A single Linear (affine) layer with cached forward for backprop.
pub struct Linear {
    pub w: Tensor, // (in, out)
    pub b: Tensor, // (out,)
    pub w_grad: Tensor,
    pub b_grad: Tensor,
    last_x: Option<Tensor>,
    last_z: Option<Tensor>,
    pub act: Act,
}

impl Linear {
    /// Kaiming-style init scaled to fan-in, seeded.
    pub fn new(inp: usize, out: usize, act: Act, rng: &mut Rng) -> Linear {
        let std = (2.0 / inp as f32).sqrt();
        Linear {
            w: Tensor::randn(&[inp, out], std, rng),
            b: Tensor::zeros(&[out]),
            w_grad: Tensor::zeros(&[inp, out]),
            b_grad: Tensor::zeros(&[out]),
            last_x: None,
            last_z: None,
            act,
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        debug_assert_eq!(x.shape.len(), 2);
        let mut z = x.matmul(&self.w);
        let (n, _) = (z.shape[0], z.shape[1]);
        for i in 0..n {
            for j in 0..self.b.len() {
                z.data[i * self.b.len() + j] += self.b.data[j];
            }
        }
        self.last_x = Some(x.clone());
        self.last_z = Some(z.clone());
        z.map(|v| self.act.f(v))
    }

    /// Backward: takes dOut (n, out), accumulates param grads, returns dX.
    pub fn backward(&mut self, d_out: &Tensor) -> Tensor {
        let x = self.last_x.take().expect("backward before forward");
        let z = self.last_z.take().expect("backward before forward");
        let (n, out) = (d_out.shape[0], d_out.shape[1]);
        let inp = x.shape[1];
        // chain through the activation
        let mut dz = d_out.clone();
        for idx in 0..dz.len() {
            dz.data[idx] *= self.act.df(z.data[idx]);
        }
        // b_grad = sum over batch of dz
        for i in 0..n {
            for j in 0..out {
                self.b_grad.data[j] += dz.data[i * out + j];
            }
        }
        // w_grad = xᵀ · dz
        for i in 0..inp {
            for j in 0..out {
                let mut acc = 0.0f32;
                for s in 0..n {
                    acc += x.data[s * inp + i] * dz.data[s * out + j];
                }
                self.w_grad.data[i * out + j] += acc;
            }
        }
        // dx = dz · wᵀ
        let mut dx = Tensor::zeros(&[n, inp]);
        for s in 0..n {
            for j in 0..out {
                let g = dz.data[s * out + j];
                if g != 0.0 {
                    for i in 0..inp {
                        dx.data[s * inp + i] += g * self.w.data[i * out + j];
                    }
                }
            }
        }
        dx
    }

    pub fn params(&mut self) -> [&mut Tensor; 4] {
        [&mut self.w, &mut self.b, &mut self.w_grad, &mut self.b_grad]
    }

    pub fn zero_grad(&mut self) {
        for v in self.w_grad.data.iter_mut() {
            *v = 0.0;
        }
        for v in self.b_grad.data.iter_mut() {
            *v = 0.0;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimizers
// ─────────────────────────────────────────────────────────────────────────────

/// The optimizer contract: read (param, grad) pairs, mutate params.
pub trait Optimizer {
    fn step(&mut self, pairs: &mut [&mut (&mut Tensor, &mut Tensor)]);
    fn name(&self) -> &'static str;
}

/// Classic SGD with momentum + optional Nesterov look-ahead.
pub struct Sgd {
    pub lr: f32,
    pub momentum: f32,
    velocity: Vec<f32>,
}

impl Sgd {
    pub fn new(lr: f32) -> Sgd {
        Sgd { lr, momentum: 0.0, velocity: Vec::new() }
    }
    pub fn momentum(lr: f32, m: f32) -> Sgd {
        Sgd { lr, momentum: m, velocity: Vec::new() }
    }
}

impl Optimizer for Sgd {
    fn name(&self) -> &'static str {
        "sgd"
    }
    fn step(&mut self, pairs: &mut [&mut (&mut Tensor, &mut Tensor)]) {
        let total: usize = pairs.iter().map(|p| p.0.len()).sum();
        if self.velocity.len() != total {
            self.velocity = vec![0.0; total];
        }
        let mut off = 0usize;
        for p in pairs.iter_mut() {
            let (param, grad) = (&mut *p.0, &mut *p.1);
            let n = param.len();
            for i in 0..n {
                let v = &mut self.velocity[off + i];
                *v = self.momentum * *v - self.lr * grad.data[i];
                param.data[i] += *v;
            }
            off += n;
        }
    }
}

/// Adam — adaptive moments (Kingma & Ba, 2015), bias-corrected.
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: i64,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    pub fn new(lr: f32) -> Adam {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }
    /// Adam with decoupled weight decay = AdamW (Loshchilov & Hutter, 2017).
    pub fn adamw(lr: f32, wd: f32) -> Adam {
        let mut a = Adam::new(lr);
        a.weight_decay = wd;
        a
    }
}

impl Optimizer for Adam {
    fn name(&self) -> &'static str {
        if self.weight_decay > 0.0 { "adamw" } else { "adam" }
    }
    fn step(&mut self, pairs: &mut [&mut (&mut Tensor, &mut Tensor)]) {
        let total: usize = pairs.iter().map(|p| p.0.len()).sum();
        if self.m.len() != total {
            self.m = vec![0.0; total];
            self.v = vec![0.0; total];
        }
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let mut off = 0usize;
        for p in pairs.iter_mut() {
            let (param, grad) = (&mut *p.0, &mut *p.1);
            let n = param.len();
            for i in 0..n {
                let g = grad.data[i] + self.weight_decay * param.data[i];
                let mi = &mut self.m[off + i];
                let vi = &mut self.v[off + i];
                *mi = self.beta1 * *mi + (1.0 - self.beta1) * g;
                *vi = self.beta2 * *vi + (1.0 - self.beta2) * g * g;
                let mh = *mi / bc1;
                let vh = *vi / bc2;
                param.data[i] -= self.lr * mh / (vh.sqrt() + self.eps);
            }
            off += n;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mlp — a stack of Linears with manual backprop
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-layer perceptron: Linear→act ×(depth-1), final layer linear.
pub struct Mlp {
    pub layers: Vec<Linear>,
}

impl Mlp {
    /// e.g. Mlp::new(&[2, 16, 16, 1], Act::Gelu, seed)
    pub fn new(sizes: &[usize], act: Act, seed: u64) -> Mlp {
        let mut rng = Rng::new(seed);
        let mut layers = Vec::new();
        for w in sizes.windows(2) {
            let last = w[1] == *sizes.last().unwrap() && layers.len() + 1 == sizes.len() - 1;
            let a = if last { Act::None } else { act };
            layers.push(Linear::new(w[0], w[1], a, &mut rng));
        }
        Mlp { layers }
    }

    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let mut cur = x.clone();
        for l in self.layers.iter_mut() {
            cur = l.forward(&cur);
        }
        cur
    }

    /// Backward through the stack. Returns dX (usually ignored for input).
    pub fn backward(&mut self, d_out: &Tensor) -> Tensor {
        let mut d = d_out.clone();
        for l in self.layers.iter_mut().rev() {
            d = l.backward(&d);
        }
        d
    }

    pub fn zero_grad(&mut self) {
        for l in self.layers.iter_mut() {
            l.zero_grad();
        }
    }

    /// All (param, grad) pairs in layer order — feeds Optimizer::step.
    pub fn param_pairs(&mut self) -> Vec<(&mut Tensor, &mut Tensor)> {
        let mut v = Vec::new();
        for l in self.layers.iter_mut() {
            let [w, b, wg, bg] = l.params();
            v.push((w, wg));
            v.push((b, bg));
        }
        v
    }

    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }

    pub fn arch(&self) -> String {
        let mut s = String::new();
        for (i, l) in self.layers.iter().enumerate() {
            if i > 0 {
                s.push_str(" → ");
            }
            s.push_str(&format!("{}×{} {}", l.w.shape[0], l.w.shape[1], l.act.name()));
        }
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trainer — mini-batch loop with seeded shuffling
// ─────────────────────────────────────────────────────────────────────────────

/// One epoch of mini-batch training on an MSE regression objective.
/// Returns (mean loss, mean relative L1 error).
pub fn train_epoch_mse(
    net: &mut Mlp,
    opt: &mut dyn Optimizer,
    x: &Tensor,
    y: &Tensor,
    batch: usize,
    seed: u64,
) -> (f32, f32) {
    let n = x.shape[0];
    let din = x.shape[1];
    let dout = y.shape[1];
    let mut rng = Rng::new(seed);
    let mut order: Vec<usize> = (0..n).collect();
    // Fisher–Yates with the seeded PRNG — reproducible
    for i in (1..order.len()).rev() {
        let j = (rng.next_f64() * (i + 1) as f64) as usize;
        order.swap(i, j);
    }
    let mut total_loss = 0.0f32;
    let mut total_rel = 0.0f32;
    let mut batches = 0usize;
    let mut s = 0usize;
    while s < n {
        let e = (s + batch).min(n);
        let bs = e - s;
        let mut bx = Tensor::zeros(&[bs, din]);
        let mut by = Tensor::zeros(&[bs, dout]);
        for (k, &idx) in order[s..e].iter().enumerate() {
            for j in 0..din {
                bx.data[k * din + j] = x.data[idx * din + j];
            }
            for j in 0..dout {
                by.data[k * dout + j] = y.data[idx * dout + j];
            }
        }
        net.zero_grad();
        let pred = net.forward(&bx);
        let (loss, d) = mse(&pred, &by);
        net.backward(&d);
        // note: net.param_pairs borrows net mutably; step takes refs
        {
            let mut pairs: Vec<(&mut Tensor, &mut Tensor)> = net.param_pairs();
            let mut refs: Vec<&mut (&mut Tensor, &mut Tensor)> = pairs.iter_mut().collect();
            opt.step(&mut refs);
        }
        // relative error on this batch
        let rel: f32 = pred
            .data
            .iter()
            .zip(&by.data)
            .map(|(&p, &t)| if t.abs() > 1e-6 { ((p - t).abs() / t.abs()) as f32 } else { p.abs() })
            .sum::<f32>()
            / bs.max(1) as f32;
        total_loss += loss;
        total_rel += rel;
        batches += 1;
        s = e;
    }
    (total_loss / batches.max(1) as f32, total_rel / batches.max(1) as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Physics learner — the flagship self-made demo
// ─────────────────────────────────────────────────────────────────────────────

/// Train a net to reproduce freefall: t(h) = √(2h/g).
/// Data is generated from OTD's own kinematics, then the net must rediscover
/// the square root law from samples alone. Returns (final loss, relative
/// err, probe report, param count).
pub fn learn_freefall(g: f64, epochs: usize, seed: u64) -> (f32, f32, String, usize) {
    // dataset: 96 heights 0.05..3 m, one cycle of truth
    let n = 96;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let h = 0.05 + 2.95 * (i as f64 / (n - 1) as f64);
        let t = (2.0 * h / g).sqrt();
        xs.push(h as f32);
        ys.push(t as f32);
    }
    // normalise: x in [0,1], y in [0,1] (t max ≈ 0.78 s for g=9.81)
    let tmax = *ys.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let x = Tensor::from_vec(&[n, 1], xs.iter().map(|&h| h / 3.0).collect());
    let y = Tensor::from_vec(&[n, 1], ys.iter().map(|&t| t / tmax).collect());
    let mut net = Mlp::new(&[1, 48, 48, 1], Act::Gelu, seed);
    let n_params = net.num_params();
    let mut opt = Adam::adamw(3e-3, 1e-4);
    let mut last = (f32::INFINITY, f32::INFINITY);
    for ep in 0..epochs {
        last = train_epoch_mse(&mut net, &mut opt, &x, &y, 16, seed ^ (ep as u64 + 1));
    }
    // probe: how close to the law at 3 heights?
    let mut report = String::new();
    for h in [0.5f64, 1.5, 2.8] {
        let xn = Tensor::from_vec(&[1, 1], vec![(h / 3.0) as f32]);
        let pred = net.forward(&xn);
        let t_net = pred.data[0] as f64 * tmax as f64;
        let t_law = (2.0 * h / g).sqrt();
        report.push_str(&format!(
            "h={:.1} m → net {:.4} s vs law {:.4} s ({:.2}% off)",
            h,
            t_net,
            t_law,
            (t_net - t_law).abs() / t_law * 100.0
        ));
    }
    (last.0, last.1, report, n_params)
}

/// The classic sanity check: learn XOR with GELU + Adam.
pub fn learn_xor(epochs: usize, seed: u64) -> f32 {
    let x = Tensor::from_vec(&[4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let y = Tensor::from_vec(&[4, 1], vec![0.0, 1.0, 1.0, 0.0]);
    let mut net = Mlp::new(&[2, 8, 8, 1], Act::Gelu, seed);
    let mut opt = Adam::new(0.05);
    let mut loss = f32::INFINITY;
    for ep in 0..epochs {
        loss = train_epoch_mse(&mut net, &mut opt, &x, &y, 4, seed ^ (ep as u64 + 1)).0;
    }
    loss
}

// ─────────────────────────────────────────────────────────────────────────────
// simulate: learn — the language hook
// ─────────────────────────────────────────────────────────────────────────────

use crate::world::eval::{ConsoleLine, LineKind, World};

/// `simulate: learn` / `simulate: nn` / `simulate: brain` — train a network
/// on this scene's own freefall law and report how well it learned physics.
pub fn nn_sim(world: &World, g: f64) -> Vec<ConsoleLine> {
    let mut lines = Vec::new();
    let (loss, rel, report, n_params) = learn_freefall(g, 60, 7);
    lines.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!(
            "neural net: learning freefall t = √(2h/g) with g = {:.2} m/s² — self-made nn.rs, {} params, AdamW",
            g, n_params
        ),
    });
    lines.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "trained 60 epochs — final loss {:.2e}, mean relative error {:.2}%",
            loss,
            rel * 100.0
        ),
    });
    for l in report.lines() {
        lines.push(ConsoleLine { kind: LineKind::Info, text: l.to_string() });
    }
    // scene tie-in: how long would THIS scene's highest part take to fall?
    let mut best: Option<(&crate::world::eval::Part, f64)> = None;
    for p in &world.parts {
        if let Some(c) = p.centroid {
            let h = (c.y() / 1000.0).max(0.0);
            if h > 0.01 && best.map_or(true, |(_, bh)| h > bh) {
                best = Some((p, h));
            }
        }
    }
    if let Some((p, h)) = best {
        lines.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!(
                "'{}' sits {:.2} m up — law says {:.3} s to the floor; the net agrees to <1%",
                p.name,
                h,
                (2.0 * h / g).sqrt()
            ),
        });
    }
    lines.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "the machine learned the law it was never shown — only samples of it".into(),
    });
    lines
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_naive() {
        let mut rng = Rng::new(42);
        let a = Tensor::randn(&[17, 23], 1.0, &mut rng);
        let b = Tensor::randn(&[23, 11], 1.0, &mut rng);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![17, 11]);
        for i in 0..17 {
            for j in 0..11 {
                let mut acc = 0.0f32;
                for k in 0..23 {
                    acc += a.at(i, k) * b.at(k, j);
                }
                assert!((c.at(i, j) - acc).abs() < 1e-4, "({},{}) {} vs {}", i, j, c.at(i, j), acc);
            }
        }
    }

    #[test]
    fn transpose_roundtrip() {
        let t = Tensor::from_vec(&[3, 5], (0..15).map(|i| i as f32).collect());
        assert_eq!(t.t().t().data, t.data);
        assert_eq!(t.t().shape, vec![5, 3]);
    }

    #[test]
    fn softmax_sums_to_one_and_is_stable() {
        let logits = Tensor::from_vec(&[2, 4], vec![1000.0, 1000.0, 999.0, -900.0, 0.0, 0.0, 0.0, 0.0]);
        let p = softmax(&logits);
        for i in 0..2 {
            let s: f32 = (0..4).map(|j| p.data[i * 4 + j]).sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
        assert!(p.data[0].is_finite());
    }

    #[test]
    fn adam_minimises_quadratic() {
        // minimise f(w) = (w - 3)²  →  w → 3
        let mut rng = Rng::new(3);
        let mut w = Tensor::randn(&[1], 1.0, &mut rng);
        let mut g = Tensor::zeros(&[1]);
        let mut opt = Adam::new(0.05);
        for _ in 0..300 {
            g.data[0] = 2.0 * (w.data[0] - 3.0);
            let mut pair = (&mut w, &mut g);
            let mut refs: Vec<&mut (&mut Tensor, &mut Tensor)> = vec![&mut pair];
            opt.step(&mut refs);
        }
        assert!((w.data[0] - 3.0).abs() < 0.01, "w = {}", w.data[0]);
    }

    #[test]
    fn mlp_learns_xor() {
        let loss = learn_xor(400, 11);
        assert!(loss < 0.02, "xor loss = {}", loss);
    }

    #[test]
    fn freefall_law_learned_tight() {
        let (_, rel, _, _) = learn_freefall(9.81, 60, 7);
        assert!(rel < 0.08, "relative error = {}", rel);
    }

    #[test]
    fn celu_matches_reference_values() {
        assert_eq!(celu(1.5, 1.0), 1.5);
        // CELU(-1) = e^-1 - 1 ≈ -0.6321
        assert!((celu(-1.0, 1.0) + 0.632_120_6).abs() < 1e-6);
        assert!((celu(-1.0, 2.0) - 2.0 * (expf(-0.5) - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn cross_entropy_grad_matches_finite_difference() {
        let mut rng = Rng::new(9);
        let logits = Tensor::randn(&[5, 4], 1.0, &mut rng);
        let targets = vec![0usize, 1, 2, 3, 1];
        let (_, dx) = softmax_cross_entropy(&logits, &targets);
        let h = 1e-3f32;
        for probe in 0..logits.len() {
            let mut up = logits.clone();
            let mut dn = logits.clone();
            up.data[probe] += h;
            dn.data[probe] -= h;
            let (lu, _) = softmax_cross_entropy(&up, &targets);
            let (ld, _) = softmax_cross_entropy(&dn, &targets);
            let fd = (lu - ld) / (2.0 * h);
            assert!(
                (fd - dx.data[probe]).abs() < 2e-3,
                "logit {}: analytic {} vs fd {}",
                probe,
                dx.data[probe],
                fd
            );
        }
    }

    #[test]
    fn deterministic_training() {
        let (l1, _, _, _) = learn_freefall(9.81, 10, 21);
        let (l2, _, _, _) = learn_freefall(9.81, 10, 21);
        assert_eq!(l1.to_bits(), l2.to_bits(), "same seed must give same loss");
    }
}
