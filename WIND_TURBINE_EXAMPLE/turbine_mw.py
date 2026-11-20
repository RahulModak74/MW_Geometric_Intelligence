# Wind-turbine operating states on a learned surface (MW recipe, toy scale).
# Solver stand-in: rotor power from the standard Cp(lambda, pitch) model. Swap in your own BEM/CFD runs.
import numpy as np, torch, pyro, pyro.distributions as dist, time, sys
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0); pyro.set_rng_seed(0)
RHO, R = 1.225, 40.0; A = np.pi*R**2
def cp(lam, beta):
    li = 1/(1/(lam+0.08*beta) - 0.035/(beta**3+1))
    return np.clip(0.5176*(116/li - 0.4*beta - 5)*np.exp(-21/li) + 0.0068*lam, 0, None)
def sample(n):
    v = np.random.uniform(4, 16, n); beta = np.random.uniform(0, 12, n)
    lam = np.random.uniform(4, 11, n); w = lam*v/R
    P = 0.5*RHO*A*cp(lam, beta)*v**3
    return np.c_[v, w, beta, P/1e6]          # 4 measurements, 3 knobs
tr, te = sample(6000), sample(1500)
mu, sd = tr.mean(0), tr.std(0)
norm = lambda x: torch.tensor((x-mu)/sd, dtype=torch.float32)
Xtr, Xte = norm(tr), norm(te)
BETZ = lambda x: 16/27*0.5*RHO*A*(x[:,0]*sd[0]+mu[0])**3/1e6   # MW upper bound per state
class Dec(torch.nn.Module):
    def __init__(s): super().__init__(); s.f=torch.nn.Sequential(torch.nn.Linear(3,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,4))
    def forward(s,z): return s.f(z)
class Enc(torch.nn.Module):
    def __init__(s): super().__init__(); s.f=torch.nn.Sequential(torch.nn.Linear(4,64),torch.nn.Tanh(),torch.nn.Linear(64,6))
    def forward(s,x): o=s.f(x); return o[:,:3], torch.nn.functional.softplus(o[:,3:])+1e-3
def run(prior_weight, steps=2000):
    pyro.clear_param_store(); dec, enc = Dec(), Enc()
    def viol(xh):   # amount by which decoded power exceeds the Betz bound at the decoded wind speed
        v=xh[:,0]*sd[0]+mu[0]; P=xh[:,3]*sd[3]+mu[3]
        return torch.relu(P-16/27*0.5*RHO*A*v**3/1e6)
    def model(x):
        pyro.module("dec",dec)
        with pyro.plate("d",x.shape[0]):
            z=pyro.sample("z",dist.Normal(torch.zeros(3),1.).to_event(1)); xh=dec(z)
            pyro.sample("x",dist.Normal(xh,0.05).to_event(1),obs=x)
            if prior_weight>0: pyro.factor("betz",-prior_weight*viol(dec(z))**2)   # the law, as one more term
    def guide(x):
        pyro.module("enc",enc)
        with pyro.plate("d",x.shape[0]):
            m,s=enc(x); pyro.sample("z",dist.Normal(m,s).to_event(1))
    svi=SVI(model,guide,Adam({"lr":2e-3}),Trace_ELBO())
    t=time.time()
    for i in range(steps):
        idx=torch.randint(0,Xtr.shape[0],(256,)); svi.step(Xtr[idx])
    train_s=time.time()-t
    with torch.no_grad():
        m,_=enc(Xte); xh=dec(m); rmse=((xh-Xte)**2).mean(0).sqrt()*torch.tensor(sd,dtype=torch.float32)
        zs=torch.randn(20000,3); xs=dec(zs); frac=(viol(xs)>0.02).float().mean().item()
        t=time.time(); dec(torch.randn(100000,3)); q=(time.time()-t)/100000*1e6
    return train_s, rmse[3].item(), frac, q
for w in (0,200):
    ts,rm,fr,q=run(w); print(f"prior_weight={w}: train {ts:.0f}s  held-out power RMSE {rm:.3f} MW  samples above Betz bound {100*fr:.2f}%  decode {q:.2f} us/state")
