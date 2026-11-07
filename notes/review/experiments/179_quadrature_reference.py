"""Issue #179: run with PYTHONPATH at each checkout to compare the quadrature before and after."""
import sys, time, warnings, numpy as np
from scipy.stats import norm
import binference as b
from binference._quadrature import bin_log_average as q
class C:
    def __init__(s,f): s.f=f; s.n=0
    def __call__(s,z,*a):
        s.n+=np.size(z); return s.f(z,*a)
print("--- issue table")
for K,hi,sg in [(1,1,1e-3),(1,1,3e-3),(1,1,1e-2),(2,2,1e-3)]:
    bins=b.Bins.linear(0,hi,K); c=C(lambda z:norm.logpdf(z,.5,sg)[None])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always"); r=q(c,bins)
    print(K,sg,np.round(np.exp(r)*hi/K,6),"evals",c.n,"warn",len(w))
print("--- very narrow, box, spike on background")
for sg in [1e-4,1e-5,1e-7]:
    c=C(lambda z:norm.logpdf(z,.5,sg)[None])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always"); r=q(c,b.Bins.linear(0,1,1))
    print("sg",sg,np.exp(r).ravel(),"evals",c.n,"warn",[str(x.message)[:50] for x in w])
c=C(lambda z:np.where(abs(z-.5)<1e-3,np.log(500),-np.inf)[None])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always"); r=q(c,b.Bins.linear(0,1,1))
print("box",np.exp(r).ravel(),"warn",[str(x.message)[:50] for x in w])
c=C(lambda z:np.logaddexp(np.log(.5)+norm.logpdf(z,.5,1e-3),np.log(.5))[None])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always"); r=q(c,b.Bins.linear(0,1,1))
print("spike+flat exact 1",np.exp(r).ravel(),"evals",c.n,"warn",len(w))
print("--- sweep: 1200 gaussians (rows=None and per-row)")
rng=np.random.default_rng(0); N=1200
mu=rng.uniform(0,1,N); s=10**rng.uniform(-2.5,-.5,N)
for K in (4,10,25):
    bins=b.Bins.linear(0,1,K); e=bins.edges[0] if hasattr(bins.edges,'__len__') and np.ndim(bins.edges[0])>0 else bins.edges
    ex=np.diff(norm.cdf((np.asarray(e)[None]-mu[:,None])/s[:,None]),axis=1)
    for name,fn in [("all",lambda: q(C0:=C(lambda z:norm.logpdf(z[None],mu[:,None],s[:,None])),bins)),]:
        pass
    c=C(lambda z:norm.logpdf(z[None],mu[:,None],s[:,None])); t=time.time()
    r=q(c,bins); dt=time.time()-t
    m=np.exp(r)/K
    with np.errstate(all='ignore'):
        rel=np.abs(m-ex).max(1)/ex.max(1)
    c2=C(lambda z,row:norm.logpdf(z,mu[row,None],s[row,None])); t=time.time()
    r2=q(c2,bins,rows=N); dt2=time.time()-t
    rel2=np.abs(np.exp(r2)/K-ex).max(1)/ex.max(1)
    print(f"K={K} shared: evals {c.n} t {dt:.2f} worst {rel.max():.2e} med {np.median(rel):.1e} | per-row: evals {c2.n} t {dt2:.2f} worst {rel2.max():.2e} med {np.median(rel2):.1e}")
print("--- 2D grid")
bins=b.Bins.grid(np.linspace(0,1,9),np.linspace(0,1,9)); m0=rng.uniform(.1,.9,(300,2)); sx=10**rng.uniform(-1.7,-.7,300)
ex=None
c=C(lambda z:norm.logpdf(z[...,0][None],m0[:,0,None],sx[:,None])+norm.logpdf(z[...,1][None],m0[:,1,None],sx[:,None])); t=time.time(); r=q(c,bins)
ee=np.linspace(0,1,9); px=np.diff(norm.cdf((ee[None]-m0[:,0,None])/sx[:,None]),axis=1); py=np.diff(norm.cdf((ee[None]-m0[:,1,None])/sx[:,None]),axis=1)
ex=np.einsum('ni,nj->nij',px,py).reshape(300,-1); rel=np.abs(np.exp(r)/64-ex).max(1)/ex.max(1)
print(f"2D shared: evals {c.n} t {time.time()-t:.2f} worst {rel.max():.2e}")
print("--- 400 cut step (rows)")
cuts=rng.uniform(0,1,400); bins=b.Bins.linear(0,1,10)
c=C(lambda z,row:np.where(z<cuts[row,None],0.,-np.inf)); t=time.time(); r=q(c,bins,rows=400,jumps=True)
print("step evals",c.n,"t",round(time.time()-t,2))
print("--- 20000 row from_density")
N=20000; mu=rng.uniform(0,1,N); s=10**rng.uniform(-2,-1,N); bins=b.Bins.linear(0,1,10)
t=time.time(); lk=b.Likelihoods.from_density(lambda z:np.exp(norm.logpdf(z[None],mu[:,None],s[:,None])),bins,kind="likelihood"); print("from_density t",round(time.time()-t,2))
