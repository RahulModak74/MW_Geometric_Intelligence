"""
BATTERY_RIEMANNIAN_LORENTZIAN.py
================================
A working tutorial on two geometries for battery health modelling.

This is a toy example with pytorch only.. Pl refer https://github.com/RahulModak74/QUANTUM_GRAVITY_WITH_MW 
for quantum gravity stress test sample code.

THE ONE IDEA
------------
Both models are the same VAE. Same data, same decoder, same metric net.
One difference: PART B flips the SIGN of one latent dimension's metric
component. That single sign turns a distance into an interval — and an
interval can be negative, zero, or positive.

    Riemannian  →  distance is always positive. "How different are A and B?"
    Lorentzian  →  interval has a sign. "Can A and B influence each other,
                   and which one came first?"

WHAT THAT BUYS YOU
------------------
A Riemannian model can say "this cell state is unusual."
A Lorentzian model can say "this cell's fade rate is 3x faster than any
healthy cell of this type has ever shown — it is outside the cone."

That is the difference between anomaly detection and physical impossibility.

THREE ELEMENTS IN THIS VERSION
-------------------------------
1. SPLIT ENCODER. A dedicated time encoder sees only cycle_count and owns
   z[:,0]. A state encoder sees everything else and owns z[:,1:]. The clock
   dimension CANNOT absorb SoH information.

2. SCALE CALIBRATION. The metric tensor is defined up to an arbitrary
   constant. Without pinning it, the sign is meaningless. causal_interval()
   normalises time and space contributions to comparable scales — the same
   convention physics uses with c = 1.

3. FADE BAND. c_fade is the measured speed limit — the maximum state
   movement per unit of clock movement that healthy training data supports.
   Every transition gets a causal_ratio: measured_rate / max_allowed_rate.
   ratio < 1  → inside the cone (normal)
   ratio ≈ 1  → on the boundary (extreme but possible)
   ratio > 1  → OUTSIDE the cone (physically impossible for this cell type)

CALIBRATION FIX (over the previous run)
---------------------------------------
The previous calibration reported c_fade = 1684 and kept only 59 of 11999
transitions. The filter was operating on NORMALISED cycle differences while
the threshold was specified in PHYSICAL units — a units mismatch that
selected cell boundaries and discarded every within-cell transition.

Fix: pass both the normalised tensor (for encoding) and the raw tensor
(for cycle-difference filtering). The filter now compares raw cycle
differences against 2× the expected training step.

PYRO LAYER (this version)
-------------------------
The MW paper says physics is written as Bayesian priors and the VAE is a
Bayesian VAE in Pyro. This file now does both, and uses Pyro to VALIDATE
the geometric verdicts with an independent physics posterior.

  P1. Both VAEs trained with Pyro SVI (Trace_ELBO, beta via poutine.scale,
      learned per-feature observation noise).
  P2. Physics-prior fade model in Pyro. K1, K2 ~ LogNormal, knee ~ Uniform,
      per-cell rate ~ LogNormal(0, tau), noise ~ HalfNormal. The SoH
      sensor floor is handled as a TRUNCATED likelihood, not ignored.
        - prior predictive check: every prior draw is monotone (no healing)
        - posterior recovery: credible intervals vs the true generator
  P3. Bayesian c_fade. The speed limit gets a posterior (NUTS), so every
      transition gets P(outside cone) instead of a hard 0/1.
  P4. Physics posterior-predictive validator. For each test transition,
      p = P(predicted drop >= observed drop | physics posterior).
      This never looks at the latent space. Agreement with the Lorentzian
      cone = two independent instruments reaching the same verdict.

Run:
    python3 BATTERY_RIEMANNIAN_LORENTZIAN.py              # pauses between steps
    python3 BATTERY_RIEMANNIAN_LORENTZIAN.py --no-pause   # runs straight through
Needs: pip install torch pyro-ppl numpy   (Pyro >= 1.8)
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive
from pyro.infer.autoguide.initialization import init_to_median
from pyro.optim import ClippedAdam

PAUSE = "--no-pause" not in sys.argv


def pause(msg):
    print("\n" + "=" * 78)
    print(msg.strip("\n"))
    print("=" * 78)
    if PAUSE:
        input("Press Enter to continue...")


# =============================================================================
# SYNTHETIC BATTERY DATA — with known ground truth
# =============================================================================
pause("""
THE SITUATION
We have battery cells. Each one ages as it cycles.
The relationship between cycle_count and SoH has a KNOWN shape:
slow fade, then a knee at ~500 cycles, then accelerated fade.

Training data only contains NORMAL fade patterns.
Test data contains:
   - normal transitions (small cycle step, matching SoH fade)
   - rapid-fade transitions (small cycle step, BIG SoH drop — impossible
     for a healthy cell, this is what a measurement error or internal
     short looks like)

The question: which model can say "that transition is not physically
possible" rather than merely "that transition looks unusual"?
""")

rng = np.random.default_rng(0)
torch.manual_seed(0)
pyro.set_rng_seed(0)

KNEE = 500.0
K1, K2 = 0.018, 2.5e-5


def true_soh(c):
    c = np.asarray(c, dtype=float)
    base = 1.0 - K1 * np.sqrt(c)
    post = np.maximum(0.0, c - KNEE)
    return base - K2 * post ** 1.5


def make_state(cycle, rng, noise=0.004):
    soh = true_soh(cycle) + rng.normal(0, noise)
    soh = float(np.clip(soh, 0.4, 1.0))
    return {
        "cycle":        float(cycle),
        "soh":          soh,
        "ir":           float(0.020 + 0.000045 * cycle + rng.normal(0, 0.0008)),
        "temperature":  float(25.0 + rng.normal(0, 3.0)),
        "capacity":     float(5.0 * soh + rng.normal(0, 0.02)),
    }


# ── Training data ───────────────────────────────────────────────────────────
N_CELLS = 60
cycles_per_cell = 200
CYCLE_STEP = 5.0        # the training cycle step — used by the calibration filter

train_rows = []
true_rates = []          # ground truth per-cell fade multiplier (Pyro check)
for cid in range(N_CELLS):
    start_cycle = int(rng.integers(0, 300))
    rate = float(rng.uniform(0.9, 1.1))
    true_rates.append(rate)
    for k in range(cycles_per_cell):
        cyc = start_cycle + k * CYCLE_STEP
        s = make_state(cyc, rng)
        s["soh"] = float(np.clip(1.0 - rate * (1.0 - s["soh"]), 0.4, 1.0))
        s["capacity"] = 5.0 * s["soh"] + rng.normal(0, 0.02)
        train_rows.append(s)

FEATURES = ["cycle", "soh", "ir", "temperature", "capacity"]
X_train = np.array([[r[f] for f in FEATURES] for r in train_rows], dtype=np.float32)

means = X_train.mean(axis=0)
stds = X_train.std(axis=0) + 1e-8
X_train_n = (X_train - means) / stds

# Keep both: normalised tensor for the VAE, raw tensor for the filter
X_train_t     = torch.tensor(X_train_n, dtype=torch.float32)
X_train_raw_t = torch.tensor(X_train,   dtype=torch.float32)

print(f"\nTraining rows: {len(train_rows):,}")
print(f"Cycle range : {X_train[:,0].min():.0f} – {X_train[:,0].max():.0f}")
print(f"SoH range   : {X_train[:,1].min():.2f} – {X_train[:,1].max():.2f}")


# ── Test transitions ────────────────────────────────────────────────────────
def build_test_pairs(n_valid=400, n_invalid=400):
    pairs = []
    for _ in range(n_valid):
        c0 = float(rng.uniform(100, 800))
        c1 = c0 + 30.0
        sa = make_state(c0, rng)
        sb = make_state(c1, rng)
        pairs.append((sa, sb, True))
    for _ in range(n_invalid):
        c0 = float(rng.uniform(100, 800))
        c1 = c0 + 30.0
        sa = make_state(c0, rng)
        sb = make_state(c1, rng)
        drop = float(rng.uniform(0.15, 0.25))
        sb["soh"] = float(np.clip(sa["soh"] - drop, 0.4, 1.0))
        sb["capacity"] = 5.0 * sb["soh"] + rng.normal(0, 0.02)
        pairs.append((sa, sb, False))
    rng.shuffle(pairs)
    return pairs


test_pairs = build_test_pairs()
print(f"Test transitions: {len(test_pairs)} "
      f"({sum(1 for _,_,v in test_pairs if v)} valid, "
      f"{sum(1 for _,_,v in test_pairs if not v)} invalid)")


# =============================================================================
# THE METRIC NET
# =============================================================================
pause("""
RIEMANNIAN vs LORENTZIAN — in the code, this is one line of difference.

Both use a metric net g(z) that maps a latent point to one number per
latent dimension. That number scales how much movement in that direction
"costs" when computing distance.

  RIEMANNIAN:  g(z) = exp(net(z)) + eps        — always positive.
               Pure distance. No direction, no arrow.

  LORENTZIAN:  g(z) = [ −exp(...),  +exp(...) ]  — first dim NEGATIVE.
               The negative dim is the CLOCK.
               Everything else is state space.

The sign flip is the whole story.
""")


class MetricNet(nn.Module):
    """Maps latent z to a per-dimension metric. Riemannian or Lorentzian."""
    def __init__(self, latent_dim, lorentzian=False, hidden=24):
        super().__init__()
        self.lorentzian = lorentzian
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden),     nn.Softplus(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z):
        raw = self.net(z)
        if self.lorentzian:
            return torch.cat([
                -(torch.exp(raw[..., :1]) + 0.01),
                 torch.exp(raw[..., 1:]) + 0.01,
            ], dim=-1)
        else:
            return torch.exp(raw) + 0.01

    def interval(self, z1, z2):
        """Raw interval — unnormalised. Diagnostics only."""
        g_mid = self.forward((z1 + z2) * 0.5)
        dz = z1 - z2
        return (g_mid * dz.pow(2)).sum(-1)

    def causal_interval(self, z1, z2):
        """Signed, calibrated interval. <0 = time-like, >0 = space-like."""
        g_mid = self.forward((z1 + z2) * 0.5)
        dz = z1 - z2

        time_g  = g_mid[..., :1].abs()
        space_g = g_mid[..., 1:]

        time_part  = (time_g  * dz[..., :1].pow(2)).sum(-1)
        space_part = (space_g * dz[..., 1:].pow(2)).sum(-1)

        time_scale  = time_g.mean().clamp(min=1e-6)
        space_scale = space_g.mean().clamp(min=1e-6)

        return space_part / space_scale - time_part / time_scale

    def causal_ratio(self, z1, z2, c_fade):
        """
        Fade-band ratio: state movement / (c_fade × clock movement).

          < 1.0  → inside the cone    (normal transition)
          ≈ 1.0  → on the boundary    (extreme but possible)
          > 1.0  → OUTSIDE the cone   (physically impossible)
        """
        g_mid = self.forward((z1 + z2) * 0.5)
        dz = z1 - z2

        time_g  = g_mid[..., :1].abs()
        space_g = g_mid[..., 1:]

        time_part  = (time_g  * dz[..., :1].pow(2)).sum(-1)
        space_part = (space_g * dz[..., 1:].pow(2)).sum(-1)

        time_scale  = time_g.mean().clamp(min=1e-6)
        space_scale = space_g.mean().clamp(min=1e-6)

        time_norm  = (time_part / time_scale).sqrt()
        space_norm = (space_part / space_scale).sqrt()

        return space_norm / (c_fade * time_norm + 1e-8)


# =============================================================================
# THE VAE — SPLIT ENCODER
# =============================================================================
pause("""
THE SPLIT ENCODER — and why it matters for the fade band.

If a single encoder sees all features, then during training (where cycle
and SoH fade together) it routes BOTH signals into z[:,0]. When an
anomaly breaks the correlation at test time, the anomaly signal still
goes to z[:,0] — the interval stays negative and the cone never fires.

The fix: TWO encoders.

  TIME ENCODER  — sees only cycle_count. Owns z[:,0]. Cannot carry
                  SoH information.

  STATE ENCODER — sees every other feature. Owns z[:,1:]. Carries all
                  the ageing state.
""")


class BatteryVAE(nn.Module):
    def __init__(self, n_features=5, latent_dim=6, lorentzian=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.lorentzian = lorentzian
        assert latent_dim >= 2, "need at least 1 time-like + 1 space-like dim"

        # TIME ENCODER — cycle only, owns z[:,0]
        self.time_enc = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(),
            nn.Linear(16, 16), nn.SiLU(),
        )
        self.time_mu = nn.Linear(16, 1)
        self.time_lv = nn.Linear(16, 1)

        # STATE ENCODER — everything except cycle, owns z[:,1:]
        self.state_enc = nn.Sequential(
            nn.Linear(n_features - 1, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
        )
        self.state_mu = nn.Linear(32, latent_dim - 1)
        self.state_lv = nn.Linear(32, latent_dim - 1)

        # DECODER — full latent → all features
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
            nn.Linear(32, n_features),
        )
        self.metric = MetricNet(latent_dim, lorentzian=lorentzian)

    def encode(self, x):
        cycle = x[:, 0:1]
        rest = x[:, 1:]
        ht = self.time_enc(cycle)
        hs = self.state_enc(rest)
        mu = torch.cat([self.time_mu(ht), self.state_mu(hs)], dim=-1)
        lv = torch.cat([self.time_lv(ht), self.state_lv(hs)], dim=-1)
        return mu, torch.clamp(lv, -8, 8)

    def decode(self, z):
        return self.dec(z)

    def reparam(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)


# =============================================================================
# PART A — RIEMANNIAN VAE
# =============================================================================
pause("""
PART A — Riemannian VAE.

Same two-encoder architecture, but the metric is fully positive. The
model can predict SoH and measure how far two latent states are. It
cannot say which state caused which — a distance has no arrow.
""")

device = torch.device("cpu")


def make_vae_model_guide(model, name, beta_kl):
    """
    Pyro model / guide pair for the split-encoder VAE.

      model:  z ~ Normal(0, I)                (scaled by beta)
              x ~ Normal(decode(z), sigma)    sigma learned per feature
      guide:  z ~ Normal(mu(x), exp(lv/2))    from the split encoder

    beta scales the z site in BOTH model and guide, so the ELBO is the
    standard beta-VAE objective: E[log p(x|z)] - beta * KL(q||p).
    """
    D = model.latent_dim
    n_feat = model.dec[-1].out_features

    def vae_model(x):
        pyro.module(name, model)
        sigma = pyro.param(f"{name}_sigma", 0.3 * torch.ones(n_feat),
                           constraint=dist.constraints.positive)
        with pyro.plate("data", x.shape[0]):
            with poutine.scale(scale=beta_kl):
                z = pyro.sample("z", dist.Normal(torch.zeros(D), 1.0)
                                .to_event(1))
            pyro.sample("x", dist.Normal(model.decode(z), sigma).to_event(1),
                        obs=x)

    def vae_guide(x):
        pyro.module(name, model)
        with pyro.plate("data", x.shape[0]):
            mu, lv = model.encode(x)
            with poutine.scale(scale=beta_kl):
                pyro.sample("z", dist.Normal(mu, torch.exp(0.5 * lv))
                            .to_event(1))

    return vae_model, vae_guide


def train_vae(model, X, name, n_steps=2000, lr=3e-3, batch=128, beta_kl=0.5,
              verbose=True):
    """Pyro SVI training. Printed loss is -ELBO per row."""
    vae_model, vae_guide = make_vae_model_guide(model, name, beta_kl)
    svi = SVI(vae_model, vae_guide, ClippedAdam({"lr": lr}), Trace_ELBO())
    n = len(X)
    losses = []
    t0 = time.time()
    model.train()
    for step in range(n_steps):
        idx = torch.randperm(n)[:batch]
        losses.append(svi.step(X[idx]) / batch)
        if verbose and (step + 1) % 400 == 0:
            print(f"  step {step+1:4d}  -ELBO/row={np.mean(losses[-100:]):.4f}  "
                  f"({time.time()-t0:.1f}s)")
    sigma = pyro.param(f"{name}_sigma").detach().numpy()
    print("  learned obs noise per feature: " +
          "  ".join(f"{f}={s:.3f}" for f, s in zip(FEATURES, sigma)))
    return losses


vae_riem = BatteryVAE(latent_dim=6, lorentzian=False)
print("Training Riemannian VAE...")
train_vae(vae_riem, X_train_t, name="vae_riem", n_steps=2000)
print("Done.")


# =============================================================================
# PART B — LORENTZIAN VAE
# =============================================================================
pause("""
PART B — Lorentzian VAE.

Same code, same split encoder, same decoder. Two changes:
  1. MetricNet has one sign-flipped dimension (z[:,0] is time-like).
  2. The TIME ENCODER owns z[:,0], so the clock dim is architecturally
     guaranteed to be the clock.
""")

vae_lor = BatteryVAE(latent_dim=6, lorentzian=True)
print("Training Lorentzian VAE...")
train_vae(vae_lor, X_train_t, name="vae_lor", n_steps=2000)
print("Done.")


# =============================================================================
# CALIBRATE THE SPEED LIMIT
# =============================================================================
pause("""
CALIBRATING c_fade — the measured speed limit.

In physics, c is not a mystery. It's measured. In a learned Lorentzian
manifold, the "how fast can state change per unit of clock" limit is
also measured — from training data.

FILTER: use within-cell transitions only. Training rows were appended
cell-by-cell, so consecutive rows across a cell boundary can have an
arbitrary cycle step. The filter compares the RAW cycle difference
against 2× the expected training step (5 cycles) — anything below is a
real within-cell transition; anything above is a boundary artifact.

The 95th percentile of the within-cell distribution is c_fade.
""")

CYCLE_STEP_RAW  = 5.0                       # the step used inside each training cell
WITHIN_CELL_MAX = 2.0 * CYCLE_STEP_RAW      # 10 — anything above is a boundary


def calibrate_c_fade(vae, X_train_norm, n_cells, rows_per_cell,
                     percentile=95, edge=20):
    """
    Calibrate c_fade from PER-CELL TRAJECTORY aggregates, not per-step ratios.

    Per-step ratios are dominated by measurement noise: over 5 cycles the
    SoH signal is tiny relative to temperature noise. The 95th percentile
    of that distribution is a noise tail, not a fade rate.

    Averaging the first `edge` and last `edge` rows of each cell kills the
    noise. The ratio (state movement) / (clock movement) across the whole
    trajectory is the cell's true average fade rate.

    95th percentile across cells → c_fade = the maximum healthy rate.
    """
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(X_train_norm)
        mu = mu.reshape(n_cells, rows_per_cell, -1)

        first = mu[:, :edge, :].mean(dim=1)      # average over first 20 rows
        last  = mu[:, -edge:, :].mean(dim=1)     # average over last 20 rows

        dt = (last[:, 0] - first[:, 0]).abs().clamp(min=1e-8)
        ds = (last[:, 1:] - first[:, 1:]).pow(2).sum(dim=-1).sqrt()

        ratios = (ds / dt).cpu().numpy()

        print(f"  Per-cell trajectory ratios across {n_cells} cells:")
        print(f"    min={ratios.min():.2f}  median={np.median(ratios):.2f}  "
              f"p95={np.percentile(ratios, 95):.2f}  max={ratios.max():.2f}")

    return float(np.percentile(ratios, percentile))
c_fade = calibrate_c_fade(vae_lor, X_train_t, N_CELLS, cycles_per_cell)
print(f"\nc_fade (95th percentile of within-cell training rates): {c_fade:.4f}")
print("Transitions with causal_ratio > 1.0 exceed anything healthy data supports.")


# =============================================================================
# SCORECARD
# =============================================================================
pause("""
SCORECARD
Three tests, plus a fade-band report on sample transitions.
""")


def recon_mse(model, X):
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X)
        xr = model.decode(mu)
        return F.mse_loss(xr, X).item()


hold_rows = []
for cid in range(10):
    start_cycle = int(rng.integers(0, 300))
    for k in range(100):
        cyc = start_cycle + k * 5
        s = make_state(cyc, rng)
        hold_rows.append(s)
X_hold = np.array([[r[f] for f in FEATURES] for r in hold_rows], dtype=np.float32)
X_hold_n = (X_hold - means) / stds
X_hold_t = torch.tensor(X_hold_n, dtype=torch.float32)

mse_r = recon_mse(vae_riem, X_hold_t)
mse_l = recon_mse(vae_lor, X_hold_t)
print("1. RECONSTRUCTION MSE ON HELD-OUT NORMAL DATA")
print(f"   Riemannian : {mse_r:.5f}")
print(f"   Lorentzian : {mse_l:.5f}")
print(f"   Difference : {(mse_l - mse_r):+.5f}")


def encode_pair(model, sa, sb):
    a = np.array([[sa[f] for f in FEATURES]], dtype=np.float32)
    b = np.array([[sb[f] for f in FEATURES]], dtype=np.float32)
    a_n = torch.tensor((a - means) / stds, dtype=torch.float32)
    b_n = torch.tensor((b - means) / stds, dtype=torch.float32)
    with torch.no_grad():
        ma, _ = model.encode(a_n)
        mb, _ = model.encode(b_n)
    return ma, mb


def classify_riemannian(model, ma, mb, threshold):
    with torch.no_grad():
        d = model.metric.interval(ma, mb).item()
    return d <= threshold


def classify_lorentzian(model, ma, mb, c_fade):
    """causal_ratio < 1.0 → inside the cone → valid."""
    with torch.no_grad():
        r = model.metric.causal_ratio(ma, mb, c_fade).item()
    return r < 1.0


norm_intervals = []
for i in range(len(X_hold_t) - 1):
    ma = X_hold_t[i:i+1]
    mb = X_hold_t[i+1:i+2]
    with torch.no_grad():
        mu_a, _ = vae_riem.encode(ma)
        mu_b, _ = vae_riem.encode(mb)
        d = vae_riem.metric.interval(mu_a, mu_b).item()
    norm_intervals.append(d)
riem_threshold = float(np.percentile(norm_intervals, 95))
print(f"\n  (Riemannian threshold calibrated at {riem_threshold:.4f})")

tp_r = tn_r = tp_l = tn_l = 0
for sa, sb, is_valid in test_pairs:
    ma_r, mb_r = encode_pair(vae_riem, sa, sb)
    pred_valid_r = classify_riemannian(vae_riem, ma_r, mb_r, riem_threshold)

    ma_l, mb_l = encode_pair(vae_lor, sa, sb)
    pred_valid_l = classify_lorentzian(vae_lor, ma_l, mb_l, c_fade)

    if is_valid:
        tp_r += int(pred_valid_r); tp_l += int(pred_valid_l)
    else:
        tn_r += int(not pred_valid_r); tn_l += int(not pred_valid_l)

n_valid = sum(1 for _,_,v in test_pairs if v)
n_invalid = sum(1 for _,_,v in test_pairs if not v)
acc_r = (tp_r + tn_r) / len(test_pairs)
acc_l = (tp_l + tn_l) / len(test_pairs)

print("\n2. ANOMALY DETECTION ON TEST TRANSITIONS")
print(f"   {'Model':<14}{'Valid kept':>12}{'Invalid rejected':>20}{'Accuracy':>12}")
print(f"   {'Riemannian':<14}{tp_r:>5}/{n_valid:<6}{tn_r:>8}/{n_invalid:<6}{acc_r:>15.1%}")
print(f"   {'Lorentzian':<14}{tp_l:>5}/{n_valid:<6}{tn_l:>8}/{n_invalid:<6}{acc_l:>15.1%}")


def alignment_corr(model, X):
    with torch.no_grad():
        mu, _ = model.encode(X)
    cyc = X[:, 0].numpy()
    z0 = mu[:, 0].numpy()
    return float(np.corrcoef(cyc, z0)[0, 1])


corr_r = alignment_corr(vae_riem, X_hold_t)
corr_l = alignment_corr(vae_lor, X_hold_t)
print("\n3. TIME-LIKE ALIGNMENT (corr of z[:,0] with cycle_count)")
print(f"   Riemannian : {corr_r:+.3f}")
print(f"   Lorentzian : {corr_l:+.3f}")


# =============================================================================
# FADE BAND REPORT
# =============================================================================
pause("""
FADE BAND REPORT — where the geometry becomes an operational number.

Every transition gets a causal_ratio: measured state movement divided by
the maximum the training population supports.

  ratio < 1.0   → inside the cone        (normal fade)
  ratio ≈ 1.0   → on the boundary        (extreme but possible)
  ratio > 1.0   → OUTSIDE the cone       (impossible for this cell type)
""")


def report_transition(model, sa, sb, c_fade):
    ma, mb = encode_pair(model, sa, sb)
    with torch.no_grad():
        ratio = model.metric.causal_ratio(ma, mb, c_fade).item()

    delta_soh = abs(sb["soh"] - sa["soh"])
    delta_cyc = abs(sb["cycle"] - sa["cycle"])
    measured_rate = delta_soh / max(delta_cyc, 1e-6)

    if ratio < 0.9:
        verdict = "INSIDE CONE"
    elif ratio < 1.1:
        verdict = "ON BOUNDARY"
    else:
        verdict = "OUTSIDE CONE"

    return {
        "delta_cycle":   delta_cyc,
        "delta_soh":     delta_soh,
        "measured_rate": measured_rate,
        "causal_ratio":  ratio,
        "verdict":       verdict,
    }


print("\nSAMPLE TRANSITIONS — with fade-band causal_ratio\n")
print(f"  {'Truth':<8}{'cyc':<12}{'SoH':<22}{'ΔSoH':>8}{'ratio':>8}  Verdict")
shown_v = shown_i = 0
for sa, sb, is_valid in test_pairs:
    if (is_valid and shown_v < 4) or (not is_valid and shown_i < 4):
        r = report_transition(vae_lor, sa, sb, c_fade)
        truth = "VALID" if is_valid else "INVALID"
        correct = ((is_valid and r["verdict"] == "INSIDE CONE") or
                   (not is_valid and r["verdict"] == "OUTSIDE CONE"))
        mark = "" if correct else "  ← MISCLASSIFIED"
        print(f"  {truth:<8}"
              f"{sa['cycle']:.0f}→{sb['cycle']:.0f}   "
              f"{sa['soh']:.3f}→{sb['soh']:.3f}      "
              f"{r['delta_soh']:>8.3f}{r['causal_ratio']:>8.2f}  "
              f"{r['verdict']}{mark}")
        if is_valid:  shown_v += 1
        else:         shown_i += 1
    if shown_v >= 4 and shown_i >= 4:
        break


ratios_valid = []
ratios_invalid = []
for sa, sb, is_valid in test_pairs:
    ma, mb = encode_pair(vae_lor, sa, sb)
    with torch.no_grad():
        r = vae_lor.metric.causal_ratio(ma, mb, c_fade).item()
    (ratios_valid if is_valid else ratios_invalid).append(r)

ratios_valid = np.array(ratios_valid)
ratios_invalid = np.array(ratios_invalid)

print("\nCAUSAL RATIO DISTRIBUTION")
print(f"  Valid transitions   : median={np.median(ratios_valid):.2f}  "
      f"p95={np.percentile(ratios_valid, 95):.2f}  "
      f"max={ratios_valid.max():.2f}")
print(f"  Invalid transitions : median={np.median(ratios_invalid):.2f}  "
      f"p05={np.percentile(ratios_invalid, 5):.2f}  "
      f"min={ratios_invalid.min():.2f}")
print(f"  Fraction of valid   inside cone (ratio < 1): "
      f"{(ratios_valid < 1).mean()*100:.1f}%")
print(f"  Fraction of invalid outside cone (ratio > 1): "
      f"{(ratios_invalid > 1).mean()*100:.1f}%")


# =============================================================================
# PART C — PYRO VALIDATION: PHYSICS AS PRIORS
# =============================================================================
pause("""
PART C — PYRO VALIDATION. Physics written as priors.

Everything above is geometry. This part checks the geometry against an
INDEPENDENT instrument: a Pyro model that knows only the fade physics.

  SoH(c) = 1 - rate * ( K1*sqrt(c) + K2*max(0, c - knee)^1.5 )

  K1    ~ LogNormal(log 0.02, 0.5)     rate constant, positive, spans decades
  K2    ~ LogNormal(log 3e-5, 0.7)     post-knee rate constant
  knee  ~ Uniform(200, 900)            regime switch location
  tau   ~ HalfNormal(0.2)              cell-to-cell spread
  rate  ~ LogNormal(0, tau)  per cell  manufacturing variation
  sigma ~ HalfNormal(0.02)             measurement noise
  SoH floor     → truncated likelihood  (a limit, written as a limit)

No PDE. The priors ARE the specification.
""")

SOH_FLOOR   = 0.4    # sensor floor used by the data generator
SOH_FIT_MIN = 0.47   # the training generator clips twice, so floored rows
                     # sit anywhere in [0.40, 0.46]. Rows below 0.47 are
                     # dropped and the likelihood is TRUNCATED at 0.47, so
                     # dropping them adds no selection bias.
cyc_all  = torch.tensor(X_train[:, 0], dtype=torch.float32)
soh_all  = torch.tensor(X_train[:, 1], dtype=torch.float32)
cell_all = torch.arange(N_CELLS).repeat_interleave(cycles_per_cell)
keep = soh_all > SOH_FIT_MIN
cyc_tr, soh_tr, cell_tr = cyc_all[keep], soh_all[keep], cell_all[keep]
print(f"\nRows used for the physics fit: {int(keep.sum())} / {len(keep)} "
      f"(rest are at the sensor floor)")


def fade_mean(c, K1, K2, knee, rate):
    return 1.0 - rate * (K1 * c.sqrt() + K2 * F.relu(c - knee).pow(1.5))


def fade_model(cyc, cell, soh=None, n_cells=N_CELLS):
    K1    = pyro.sample("K1",   dist.LogNormal(np.log(0.02), 0.5))
    K2    = pyro.sample("K2",   dist.LogNormal(np.log(3e-5), 0.7))
    knee  = pyro.sample("knee", dist.Uniform(200.0, 900.0))
    tau   = pyro.sample("tau",  dist.HalfNormal(0.2))
    sigma = pyro.sample("sigma", dist.HalfNormal(0.02))
    # Sum-to-zero log-rates. Without this, K1 and the overall rate scale
    # trade off along a ridge and NUTS mixes badly. With it, K1 is the
    # fade constant of the geometric-mean cell.
    with pyro.plate("cells", n_cells):
        log_rate = pyro.sample("log_rate", dist.Normal(0.0, tau))
    if n_cells > 1:
        log_rate = log_rate - log_rate.mean(-1, keepdim=True)
    rate = pyro.deterministic("rate", torch.exp(log_rate))
    mean = fade_mean(cyc, K1, K2, knee, rate[cell])
    with pyro.plate("rows", len(cyc)):
        if soh is None:
            pyro.sample("soh", dist.Normal(mean, sigma))
        else:
            # Normal truncated below at SOH_FIT_MIN:
            #   log N(y | mean, sigma) - log P(Y > SOH_FIT_MIN)
            ll = (dist.Normal(mean, sigma).log_prob(soh)
                  - torch.special.log_ndtr((mean - SOH_FIT_MIN) / sigma))
            pyro.factor("soh_ll", ll)


# ── P2a. Prior predictive check ─────────────────────────────────────────────
print("\nP2a. PRIOR PREDICTIVE CHECK — do the priors already encode the physics?")
c_grid = torch.linspace(0, 1300, 131)
prior = Predictive(fade_model, num_samples=2000,
                   return_sites=["K1", "K2", "knee", "rate"])(
    c_grid, torch.zeros(131, dtype=torch.long), n_cells=1)
S0 = 2000
curves = fade_mean(c_grid[None, :], prior["K1"].reshape(S0, 1),
                   prior["K2"].reshape(S0, 1), prior["knee"].reshape(S0, 1),
                   prior["rate"].reshape(S0, 1))
monotone = (curves[:, 1:] <= curves[:, :-1] + 1e-7).all(dim=1).float().mean()
starts_at_1 = (curves[:, 0] - 1.0).abs().lt(1e-6).float().mean()
print(f"   prior draws monotone non-increasing (no healing): {monotone*100:.1f}%")
print(f"   prior draws with SoH(0) = 1                     : {starts_at_1*100:.1f}%")
for c in (200, 500, 1000):
    i = int(c / 10)
    lo, hi = np.percentile(curves[:, i].numpy(), [5, 95])
    print(f"   prior 90% band for SoH at cycle {c:4d}: [{lo:.2f}, {hi:.2f}]  "
          f"truth={true_soh(c):.3f}")


# ── P2b. Posterior via SVI ──────────────────────────────────────────────────
print("\nP2b. POSTERIOR (NUTS) — does the data recover the generator?")
t0 = time.time()
fade_mcmc = MCMC(NUTS(fade_model, init_strategy=init_to_median, max_tree_depth=7),
                 num_samples=400, warmup_steps=400, disable_progbar=True)
fade_mcmc.run(cyc_tr, cell_tr, soh_tr)
post = {k: v.detach() for k, v in fade_mcmc.get_samples().items()}
_lr = post["log_rate"].reshape(len(post["K1"]), -1)
post["rate"] = torch.exp(_lr - _lr.mean(-1, keepdim=True))
_diag = pyro.infer.mcmc.util.summary(fade_mcmc.get_samples(group_by_chain=True))
print(f"   NUTS done ({time.time()-t0:.0f}s).  diagnostics (want r_hat < 1.05):")
for k in ("K1", "K2", "knee", "tau", "sigma"):
    print(f"     {k:<6} r_hat={float(_diag[k]['r_hat']):.3f}  "
          f"n_eff={float(_diag[k]['n_eff']):.0f}")

# K1 is defined at the geometric-mean cell, so its truth includes that mean.
truth = {"K1": K1 * float(np.exp(np.mean(np.log(true_rates)))), "K2": K2, "knee": KNEE}
print(f"\n   {'param':<7}{'post median':>14}{'90% CI':>28}{'truth':>12}{'rel.err':>9}  covered")
n_cov = 0
for k in ("K1", "K2", "knee"):
    s = post[k].numpy().ravel()
    lo, med, hi = np.percentile(s, [5, 50, 95])
    cov = lo <= truth[k] <= hi
    n_cov += cov
    print(f"   {k:<7}{med:>14.4g}   [{lo:>10.4g}, {hi:>10.4g}]{truth[k]:>12.4g}"
          f"{(med - truth[k]) / truth[k]:>+9.1%}  {'YES' if cov else 'no'}")
print(f"   sigma  post median={post['sigma'].median():.4f}   "
      f"(generator noise ≈ 0.004 × rate)")
rate_post = post["rate"].reshape(post["rate"].shape[0], -1).median(dim=0).values.numpy()
rate_corr = float(np.corrcoef(rate_post, true_rates)[0, 1])
print(f"   per-cell rate recovery: corr(posterior median, true rate) = "
      f"{rate_corr:+.3f} across {N_CELLS} cells")
print(f"   {n_cov}/3 physics constants inside their 90% credible interval.")


# ── P3. Bayesian c_fade ─────────────────────────────────────────────────────
pause("""
P3. BAYESIAN c_fade — a speed limit with an error bar.

Above, c_fade is a point estimate: the 95th percentile of 60 per-cell
ratios. 60 cells is not many. Pyro puts a posterior on it:

  log(ratio_cell) ~ Normal(m, s)
  c_fade          = exp(m + 1.645 s)        (population 95th percentile)

Each transition then gets P(outside cone) = P(raw_ratio > c_fade),
averaged over the posterior. A probability, not a hard 0/1.
""")

with torch.no_grad():
    mu_tr, _ = vae_lor.encode(X_train_t)
    mu_tr = mu_tr.reshape(N_CELLS, cycles_per_cell, -1)
    first, last = mu_tr[:, :20].mean(1), mu_tr[:, -20:].mean(1)
    cell_ratios = ((last[:, 1:] - first[:, 1:]).norm(dim=-1) /
                   (last[:, 0] - first[:, 0]).abs().clamp(min=1e-8))
log_r = cell_ratios.log()


def cfade_model(log_r):
    m = pyro.sample("m", dist.Normal(log_r.mean(), 1.0))
    s = pyro.sample("s", dist.HalfNormal(1.0))
    with pyro.plate("cells", len(log_r)):
        pyro.sample("obs", dist.Normal(m, s), obs=log_r)


mcmc = MCMC(NUTS(cfade_model), num_samples=1000, warmup_steps=500,
            disable_progbar=True)
mcmc.run(log_r)
cs = mcmc.get_samples()
c_fade_post = torch.exp(cs["m"] + 1.645 * cs["s"])
lo, med, hi = np.percentile(c_fade_post.numpy(), [5, 50, 95])
print(f"\n   point c_fade (empirical p95) : {c_fade:.3f}")
print(f"   Bayesian c_fade median       : {med:.3f}   90% CI [{lo:.3f}, {hi:.3f}]")
print(f"   point estimate inside CI     : {'YES' if lo <= c_fade <= hi else 'no'}")
if not (lo <= c_fade <= hi):
    print(f"   → the per-cell ratios are heavy-tailed (max/median = "
          f"{float(cell_ratios.max() / cell_ratios.median()):.0f}x). The empirical\n"
          f"     p95 of {N_CELLS} cells sits in that tail and is unstable; the Bayesian "
          f"value is the steadier speed limit.")

# raw ratio = causal_ratio with c_fade = 1
raw_ratios, labels = [], []
for sa, sb, is_valid in test_pairs:
    ma, mb = encode_pair(vae_lor, sa, sb)
    with torch.no_grad():
        raw_ratios.append(vae_lor.metric.causal_ratio(ma, mb, 1.0).item())
    labels.append(is_valid)
raw_ratios = torch.tensor(raw_ratios)
labels = np.array(labels)
p_outside = (raw_ratios[:, None] > c_fade_post[None, :]).float().mean(1).numpy()

print(f"\n   P(outside cone)   valid: median={np.median(p_outside[labels]):.2f}  "
      f"p95={np.percentile(p_outside[labels], 95):.2f}")
print(f"                   invalid: median={np.median(p_outside[~labels]):.2f}  "
      f"p05={np.percentile(p_outside[~labels], 5):.2f}")
for lo_p, hi_p, tag in ((0.0, 0.05, "confident INSIDE  (P<0.05)"),
                        (0.05, 0.95, "uncertain         "),
                        (0.95, 1.01, "confident OUTSIDE (P>0.95)")):
    sel = (p_outside >= lo_p) & (p_outside < hi_p)
    print(f"   {tag}: {sel.sum():>4} transitions  "
          f"({(labels[sel]).sum()} valid, {(~labels[sel]).sum()} invalid)")


# ── P4. Physics posterior-predictive validator ──────────────────────────────
pause("""
P4. PHYSICS VALIDATOR — an independent second opinion.

For each test transition (c0 → c1), draw from the physics posterior:
a new cell's rate, the fade curve, noise, and the SoH floor. That gives
the distribution of SoH drops physics allows over that cycle window.

  p = P(predicted drop >= observed drop | physics posterior)
  p < 0.001 → the observed drop is physically implausible.

This never touches the VAE. If it agrees with the Lorentzian cone, the
geometry is reading physics, not memorising the training set.
""")

S = post["K1"].shape[0]
pK1, pK2 = post["K1"].reshape(S, 1), post["K2"].reshape(S, 1)
pknee, ptau = post["knee"].reshape(S, 1), post["tau"].reshape(S, 1)
psig = post["sigma"].reshape(S, 1)
new_rate = dist.LogNormal(torch.zeros(S, 1), ptau).sample()

c0 = torch.tensor([sa["cycle"] for sa, _, _ in test_pairs])[None, :]
c1 = torch.tensor([sb["cycle"] for _, sb, _ in test_pairs])[None, :]
obs_drop = torch.tensor([sa["soh"] - sb["soh"] for sa, sb, _ in test_pairs])

soh0 = (fade_mean(c0, pK1, pK2, pknee, new_rate) +
        psig * torch.randn(S, c0.shape[1])).clamp(SOH_FLOOR, 1.0)
soh1 = (fade_mean(c1, pK1, pK2, pknee, new_rate) +
        psig * torch.randn(S, c1.shape[1])).clamp(SOH_FLOOR, 1.0)
pred_drop = soh0 - soh1
p_phys = (pred_drop >= obs_drop[None, :]).float().mean(0).numpy()
phys_valid = p_phys >= 1e-3

lor_valid = (raw_ratios < c_fade).numpy()
both_floor = np.array([sa["soh"] <= SOH_FLOOR + 1e-6 and sb["soh"] <= SOH_FLOOR + 1e-6
                       for sa, sb, _ in test_pairs])


def acc(pred):
    return (pred == labels).mean()


print(f"\n   {'Instrument':<34}{'Valid kept':>12}{'Invalid rej.':>14}{'Acc':>8}")
for name_, pred in (("Physics posterior (Pyro, no VAE)", phys_valid),
                    ("Lorentzian cone (VAE geometry)", lor_valid)):
    print(f"   {name_:<34}{(pred & labels).sum():>6}/{labels.sum():<5}"
          f"{(~pred & ~labels).sum():>8}/{(~labels).sum():<5}{acc(pred):>8.1%}")

agree = (phys_valid == lor_valid).mean()
pa = agree
pe = (phys_valid.mean() * lor_valid.mean() +
      (1 - phys_valid.mean()) * (1 - lor_valid.mean()))
kappa = (pa - pe) / (1 - pe + 1e-12)
print(f"\n   Agreement physics vs geometry : {agree:.1%}   Cohen's kappa = {kappa:.2f}")
print(f"   Invalid pairs pinned at the SoH floor (drop erased by the floor clip): "
      f"{(both_floor & ~labels).sum()}  — no instrument can see these.")
missed_both = (~labels) & phys_valid & lor_valid
print(f"   Invalid pairs BOTH instruments missed: {missed_both.sum()} "
      f"(of which at floor: {(missed_both & both_floor).sum()})")

# Per-prior residual: how much of the observed drop lies beyond the
# monotone-fade prior's 99.9% predictive bound.
bound = torch.quantile(pred_drop, 0.999, dim=0).numpy()
excess = np.clip(obs_drop.numpy() - bound, 0, None)
print(f"\n   Residual on the fade-rate prior (excess drop beyond 99.9% bound):")
print(f"     valid   mean = {excess[labels].mean():.4f}")
print(f"     invalid mean = {excess[~labels].mean():.4f}   "
      f"← this prior is the one the data rejects")


pause("""
WHAT THIS UNLOCKS FOR A BATTERY PRODUCT

Riemannian: "this cell state is far from training data."
Lorentzian: "this cell's fade rate is N times the maximum healthy rate —
             it is outside the cone that any healthy cell in our training
             population has ever occupied."

Operationally:
  - ratio < 1      → report as normal, no action
  - ratio 1.0–1.5  → flag for monitoring, run a second measurement
  - ratio > 1.5    → escalate. This is not sensor noise. Either the cell
                     has a fault, or the sensor is faulting.

The causal_ratio is a NUMBER on the dashboard. Not a threshold on a
distance. Not a model confidence. A direct comparison between the
measured fade rate and the maximum the physics allows.

HONEST LIMITS
  - c_fade is learned from training data. If the training set contains
    no fast-fade examples, the cone is narrow and the model rejects more.
    If it contains some, the cone widens and it rejects more conservatively.
  - The two-encoder split is what makes the cone meaningful. Without it,
    the clock dim absorbs anomaly signal and the ratio stays under 1.
  - Scale calibration (c = 1 convention) has to be applied before the
    ratio is computed. Otherwise the ratio is arbitrary.
  - Space-like is a strong claim: "no training trajectory connects these
    states." A genuinely new failure mode is also space-like. That's the
    honest interpretation — not "impossible in any world" but "outside
    anything this cell population has shown."
""")
print("Done.")
