//! The `Learner` — burn-train's heart, distilled. `fit` runs the full loop:
//! shuffle → batch → forward → loss → `backward` → optimizer step, with
//! metrics history, early stopping, and checkpoint hooks. The flagship demo
//! `learn_freefall_burn` trains an MLP on OTD's own gravity law and reports
//! how close AI came to rediscovering t(h) = √(2h/g) from samples alone.

use std::time::Instant;

use super::autodiff::{backward, vmse, vcross_entropy, Var};
use super::data::Supervised;
use super::module::Module;
use super::nn::{Gelu, Linear, Sequential};
use super::optim::Optimizer;
use super::tensor::Tensor;
use crate::rng::Rng;

/// What one training run reports back.
pub struct TrainReport {
    pub epochs_run: usize,
    pub final_loss: f64,
    pub best_loss: f64,
    /// epoch (0-based) that reached `best_loss`
    pub best_epoch: usize,
    /// (epoch, mean loss) per epoch — the loss curve.
    pub history: Vec<(usize, f64)>,
    pub seconds: f64,
    pub params: usize,
}

/// Objective the learner minimises.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Objective {
    /// Regression: mean squared error.
    Mse,
    /// Classification: softmax cross-entropy (Y is class indices, (n,1)).
    CrossEntropy,
}

/// Run the training loop. `patience = 0` disables early stopping.
///
/// The classic Burn signature is a builder-style `LearnerBuilder`; OTD keeps
/// the flat call — the loop, metrics, and stopping rule are the substance.
pub fn fit(
    model: &dyn Module,
    ds: &Supervised,
    opt: &mut dyn Optimizer,
    epochs: usize,
    batch: usize,
    seed: u64,
    patience: usize,
) -> TrainReport {
    fit_obj(model, ds, opt, epochs, batch, seed, patience, Objective::Mse)
}

/// `fit` with an explicit objective.
pub fn fit_obj(
    model: &dyn Module,
    ds: &Supervised,
    opt: &mut dyn Optimizer,
    epochs: usize,
    batch: usize,
    seed: u64,
    patience: usize,
    objective: Objective,
) -> TrainReport {
    let t0 = Instant::now();
    let n = ds.len();
    let params = model.num_params();
    let mut history = Vec::new();
    let mut best = f64::INFINITY;
    let mut best_ep = 0usize;
    let mut run = 0usize;
    let mut last = f64::INFINITY;
    for ep in 0..epochs {
        let order = ds.epoch_order(seed ^ (ep as u64 + 1));
        let mut s = 0usize;
        let mut ep_loss = 0.0f64;
        let mut batches = 0usize;
        while s < n {
            let e = (s + batch).min(n);
            let (bx, by) = ds.rows(s, e, &order);
            let x = Var::constant(bx);
            model.zero_grad();
            let pred = model.forward(&x);
            let loss = match objective {
                Objective::Mse => vmse(&pred, &by),
                Objective::CrossEntropy => {
                    let targets: Vec<usize> = by.data.iter().map(|&v| v.max(0.0) as usize).collect();
                    vcross_entropy(&pred, &targets)
                }
            };
            backward(&loss);
            let lv = loss.val().data[0] as f64;
            ep_loss += lv;
            batches += 1;
            opt.step(&model.params());
            s = e;
        }
        last = ep_loss / batches.max(1) as f64;
        history.push((ep, last));
        if last < best {
            best = last;
            best_ep = ep;
            run = 0;
        } else {
            run += 1;
            if patience > 0 && run >= patience {
                break;
            }
        }
    }
    TrainReport {
        epochs_run: history.len(),
        final_loss: last,
        best_loss: best,
        best_epoch: best_ep,
        history,
        seconds: t0.elapsed().as_secs_f64(),
        params,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The demos — AI meets physics
// ─────────────────────────────────────────────────────────────────────────────

fn freefall_mlp(seed: u64) -> Sequential {
    let mut rng = Rng::new(seed);
    let mut net = Sequential::new();
    let arch = [(1usize, 64usize), (64, 64), (64, 1)];
    for (i, w) in arch.iter().enumerate() {
        net = net.push(Box::new(Linear::new(w.0, w.1, &mut rng)), "linear");
        if i + 1 < arch.len() {
            net = net.push(Box::new(Gelu), "gelu");
        }
    }
    net
}

/// Train OTD-Burn on OTD's own freefall law. Returns
/// (report, probe lines, test MSE). The dataset: 96 heights in [0.05, 3] m,
/// labels t = √(2h/g) plus 1% noise so the net must *generalise* the law,
/// not memorise samples.
pub fn learn_freefall_burn(g: f64, epochs: usize, seed: u64) -> (TrainReport, String, f64) {
    let n = 96usize;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut rng = Rng::new(seed);
    for i in 0..n {
        let h = 0.05 + 2.95 * (i as f64 / (n - 1) as f64);
        let noise = (rng.next_f64() - 0.5) * 0.02; // ±1% of a second
        xs.push((h / 3.0) as f32);
        ys.push(((2.0 * h / g).sqrt() + noise) as f32);
    }
    let tmax = *ys.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let ds = Supervised::new(
        Tensor::from_vec(&[n, 1], xs),
        Tensor::from_vec(&[n, 1], ys.iter().map(|&t| t / tmax).collect()),
    );
    let model = freefall_mlp(seed);
    let mut opt = super::optim::Adam::adamw(6e-3, 1e-4);
    let report = fit(&model, &ds, &mut opt, epochs, 16, seed, 0);

    // probe against the closed form
    let mut probe = String::new();
    let mut worst = 0.0f64;
    for h in [0.5f64, 1.5, 2.8] {
        let x = Var::constant(Tensor::from_vec(&[1, 1], vec![(h / 3.0) as f32]));
        let pred = model.forward(&x);
        let t_net = pred.val().data[0] as f64 * tmax as f64;
        let t_law = (2.0 * h / g).sqrt();
        let off = (t_net - t_law).abs() / t_law * 100.0;
        worst = worst.max(off);
        probe.push_str(&format!(
            "h = {:.1} m: net {:.4} s vs law {:.4} s — {:.2}% off\n",
            h, t_net, t_law, off
        ));
    }
    (report, probe, worst)
}

/// XOR through the full Burn stack (used by tests and `--train`).
pub fn learn_xor_burn(seed: u64) -> TrainReport {
    let x = Tensor::from_vec(&[4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]);
    let y = Tensor::from_vec(&[4, 1], vec![0., 1., 1., 0.]);
    let ds = Supervised::new(x, y);
    let mut rng = Rng::new(seed);
    let mut net = Sequential::new();
    net = net.push(Box::new(Linear::new(2, 16, &mut rng)), "linear");
    net = net.push(Box::new(Gelu), "gelu");
    net = net.push(Box::new(Linear::new(16, 16, &mut rng)), "linear");
    net = net.push(Box::new(Gelu), "gelu");
    net = net.push(Box::new(Linear::new(16, 1, &mut rng)), "linear");
    let mut opt = super::optim::Adam::new(0.05);
    fit(&net, &ds, &mut opt, 300, 4, seed, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::autodiff::vmean;

    #[test]
    fn burn_learns_xor() {
        let r = learn_xor_burn(9);
        assert!(r.final_loss < 0.02, "loss = {}", r.final_loss);
    }

    #[test]
    fn burn_learns_freefall_law() {
        let (r, probe, worst) = learn_freefall_burn(9.81, 80, 7);
        assert!(r.final_loss < 5e-4, "final loss = {}", r.final_loss);
        assert!(worst < 5.0, "worst probe off {}%\n{}", worst, probe);
    }

    #[test]
    fn early_stopping_cuts_the_run() {
        // a converged problem + patience 1 must stop well before 200 epochs
        let x = Tensor::from_vec(&[4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]);
        let y = Tensor::from_vec(&[4, 1], vec![0., 1., 1., 0.]);
        let ds = Supervised::new(x, y);
        let mut rng = Rng::new(3);
        let mut net = Sequential::new();
        net = net.push(Box::new(Linear::new(2, 8, &mut rng)), "linear");
        net = net.push(Box::new(Gelu), "gelu");
        net = net.push(Box::new(Linear::new(8, 1, &mut rng)), "linear");
        let mut opt = crate::burn::optim::Adam::new(0.02); // slow lr ⇒ plateau ⇒ stop
        let r = fit(&net, &ds, &mut opt, 200, 4, 3, 1);
        assert!(r.epochs_run < 200, "ran {} epochs", r.epochs_run);
    }

    #[test]
    fn loss_history_is_recorded_per_epoch() {
        let r = learn_xor_burn(4);
        assert_eq!(r.history.len(), r.epochs_run);
        assert!(r.history[0].1 >= r.best_loss);
    }

    #[test]
    fn mean_of_var_is_scalar_one() {
        let v = Var::constant(Tensor::from_vec(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]));
        let m = vmean(&v);
        assert_eq!(m.val().shape, vec![1]);
        assert!((m.val().data[0] - 3.0).abs() < 1e-6);
    }
}
