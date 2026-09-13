//! Optimizers and LR schedules — the burn-optim analogue. All of them mutate
//! `Var` leaves through their `Rc` handles, which is why `step` takes only
//! `&[Var]`: interior mutability is the whole trick.

use super::autodiff::Var;
use super::backend::NdArray;
use super::tensor::Tensor;

/// The optimizer contract (Burn's `Optimizer` trait, trimmed).
pub trait Optimizer {
    /// One update step over the module's parameters (order must be stable).
    fn step(&mut self, params: &[Var]);
    /// Set the learning rate (the LR scheduler calls this every epoch).
    fn set_lr(&mut self, lr: f64);
    fn name(&self) -> &'static str;
}

/// SGD with optional momentum.
pub struct Sgd {
    pub lr: f64,
    pub momentum: f64,
    velocity: Vec<Tensor<NdArray>>,
}

impl Sgd {
    pub fn new(lr: f64) -> Sgd {
        Sgd { lr, momentum: 0.0, velocity: Vec::new() }
    }
    pub fn momentum(lr: f64, m: f64) -> Sgd {
        Sgd { lr, momentum: m, velocity: Vec::new() }
    }
}

impl Optimizer for Sgd {
    fn name(&self) -> &'static str {
        "sgd"
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    fn step(&mut self, params: &[Var]) {
        if self.velocity.len() != params.len() {
            self.velocity = params.iter().map(|p| p.val().scale(0.0)).collect();
        }
        for (i, p) in params.iter().enumerate() {
            let v = p.val();
            let g = p.grad();
            let mut upd = Vec::with_capacity(v.len());
            let vel = &mut self.velocity[i];
            for j in 0..v.len() {
                let nv = self.momentum * vel.data[j] as f64 - self.lr * g.data[j] as f64;
                vel.data[j] = nv as f32;
                upd.push((v.data[j] as f64 + nv) as f32);
            }
            p.set_value(Tensor::from_vec(&v.shape.clone(), upd));
        }
    }
}

/// Adam (Kingma & Ba, 2015) — bias-corrected adaptive moments.
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    t: i64,
    m: Vec<Tensor<NdArray>>,
    v: Vec<Tensor<NdArray>>,
}

impl Adam {
    pub fn new(lr: f64) -> Adam {
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

    /// AdamW — decoupled weight decay (Loshchilov & Hutter, 2017).
    pub fn adamw(lr: f64, wd: f64) -> Adam {
        let mut a = Adam::new(lr);
        a.weight_decay = wd;
        a
    }
}

impl Optimizer for Adam {
    fn name(&self) -> &'static str {
        if self.weight_decay > 0.0 { "adamw" } else { "adam" }
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    fn step(&mut self, params: &[Var]) {
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| p.val().scale(0.0)).collect();
            self.v = params.iter().map(|p| p.val().scale(0.0)).collect();
        }
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (i, p) in params.iter().enumerate() {
            let v = p.val();
            let g = p.grad();
            let mi = &mut self.m[i];
            let vi = &mut self.v[i];
            let mut upd = Vec::with_capacity(v.len());
            for j in 0..v.len() {
                let gj = g.data[j] as f64 + self.weight_decay * v.data[j] as f64;
                mi.data[j] = (self.beta1 * mi.data[j] as f64 + (1.0 - self.beta1) * gj) as f32;
                vi.data[j] = (self.beta2 * vi.data[j] as f64 + (1.0 - self.beta2) * gj * gj) as f32;
                let mh = mi.data[j] as f64 / bc1;
                let vh = vi.data[j] as f64 / bc2;
                upd.push((v.data[j] as f64 - self.lr * mh / (vh.sqrt() + self.eps)) as f32);
            }
            p.set_value(Tensor::from_vec(&v.shape.clone(), upd));
        }
    }
}

/// Learning-rate schedules over epochs (Burn's scheduler trio).
#[derive(Clone, Copy, Debug)]
pub enum LrScheduler {
    Constant(f64),
    /// Linear decay from start to end across the run.
    Linear { start: f64, end: f64 },
    /// Cosine annealing from start to end (restart-free).
    Cosine { start: f64, end: f64 },
}

impl LrScheduler {
    /// LR for `epoch` (0-based) of `total` epochs.
    pub fn at(&self, epoch: usize, total: usize) -> f64 {
        let t = if total <= 1 { 0.0 } else { epoch as f64 / (total - 1) as f64 };
        match self {
            LrScheduler::Constant(lr) => *lr,
            LrScheduler::Linear { start, end } => start + (end - start) * t,
            LrScheduler::Cosine { start, end } => {
                let c = (core::f64::consts::PI * t).cos();
                end + (start - end) * 0.5 * (1.0 + c)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_minimises_quadratic_through_vars() {
        let w = Var::param(Tensor::<NdArray>::from_vec(&[1], vec![5.0]));
        let mut opt = Adam::new(0.1);
        for _ in 0..200 {
            // f(w) = (w-2)²  →  grad = 2(w-2); rebuild a tiny graph per step
            let v = w.val();
            let g = Tensor::<NdArray>::from_vec(&[1], vec![2.0 * (v.data[0] - 2.0)]);
            // write the analytic grad straight into the leaf
            w.set_value(Tensor::<NdArray>::from_vec(&[1], vec![v.data[0]]));
            w.node_grad(g);
            opt.step(&[w.clone()]);
        }
        assert!((w.val().data[0] - 2.0).abs() < 0.01, "w={}", w.val().data[0]);
    }

    #[test]
    fn sgd_momentum_converges() {
        let w = Var::param(Tensor::<NdArray>::from_vec(&[1], vec![-4.0]));
        let mut opt = Sgd::momentum(0.05, 0.9);
        for _ in 0..300 {
            let v = w.val();
            let g = Tensor::<NdArray>::from_vec(&[1], vec![2.0 * (v.data[0] - 1.0)]);
            w.node_grad(g);
            opt.step(&[w.clone()]);
        }
        assert!((w.val().data[0] - 1.0).abs() < 0.02, "w={}", w.val().data[0]);
    }

    #[test]
    fn schedulers_start_and_end_correctly() {
        let s = LrScheduler::Linear { start: 0.1, end: 0.01 };
        assert!((s.at(0, 10) - 0.1).abs() < 1e-12);
        assert!((s.at(9, 10) - 0.01).abs() < 1e-12);
        let c = LrScheduler::Cosine { start: 0.1, end: 0.01 };
        assert!((c.at(0, 10) - 0.1).abs() < 1e-12);
        assert!((c.at(9, 10) - 0.01).abs() < 1e-12);
        assert!(c.at(5, 10) < 0.1 && c.at(5, 10) > 0.01);
        assert_eq!(LrScheduler::Constant(0.3).at(7, 10), 0.3);
    }
}
