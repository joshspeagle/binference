"""SBC of LogisticGP with inferred amplitude and length scale.

Arguments: case n_simulations workers output_directory. Run with OPENBLAS_NUM_THREADS=1.
"""
import sys, time, warnings, multiprocessing as mp, numpy as np
import binference
from binference import Bins, priors as P
from binference.simulate import prior_predictive
CASES={
 "2d16":(lambda:Bins.grid(np.linspace(0,1,17),np.linspace(0,1,17)),lambda:P.LogisticGP(n_basis=10),1000,0.03,dict(n_chains=2,n_warmup=300,n_samples=300)),
 "1d":(lambda:Bins.linear(0,1,8),P.LogisticGP,60,0.1,dict(n_chains=2,n_warmup=400,n_samples=300)),
 "2d":(lambda:Bins.grid(np.linspace(0,1,7),np.linspace(0,1,7)),lambda:P.LogisticGP(),300,0.05,dict(n_chains=2,n_warmup=400,n_samples=300)),
}
NDRAW=19
def one(a):
    case,seed=a; warnings.simplefilter("ignore")
    mk,prior,N,noise,kw=CASES[case]; bins=mk(); pr=prior()
    rng=np.random.default_rng([129,seed])
    if bins.ndim==1:
        lk,truth=prior_predictive(pr,bins,N,noise=noise)(rng)
    else:
        truth=pr.bind(bins).sample(1,rng)[0]
        cell=rng.choice(bins.K,size=N,p=truth/truth.sum())
        ij=np.stack(bins.cell_index,axis=-1)[cell]
        ex,ey=bins.edges
        z=np.stack([ex[ij[:,0]]+rng.uniform(0,1,N)*np.diff(ex)[ij[:,0]],ey[ij[:,1]]+rng.uniform(0,1,N)*np.diff(ey)[ij[:,1]]],axis=-1)
        obs=z+noise*rng.standard_normal((N,2))
        lk=binference.Likelihoods.from_gaussian(obs,noise,bins)
    t=time.perf_counter(); r=binference.infer(lk,pr,method="nuts",rng=rng,check=False,**kw); dt=time.perf_counter()-t
    d=r.samples; idx=np.linspace(0,len(d)-1,NDRAW).astype(int); d=d[idx]
    rank=(d<truth).sum(0)
    return rank,dt,float(np.min(r.diagnostics["ess_bulk"])),int(r.diagnostics.get("n_divergent",0)),float(np.nanmin(r.diagnostics.get("ebfmi",[np.nan])))
if __name__=="__main__":
    case,n,w=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
    with mp.get_context("spawn").Pool(w) as p: res=p.map(one,[(case,s) for s in range(n)],chunksize=1)
    R=np.array([r[0] for r in res]); t=np.array([r[1] for r in res]); e=np.array([r[2] for r in res])
    from scipy import stats
    H=np.array([np.bincount(R[:,k],minlength=NDRAW+1) for k in range(R.shape[1])])
    H5=H.reshape(H.shape[0],5,4).sum(2)
    p=stats.chisquare(H5,axis=1).pvalue
    print(case,"n",n,"time total",t.sum().round(1),"max",t.max().round(1),"minESS",e.min().round(0),"medESS",np.median(e).round(0),"div",sum(r[3] for r in res),"ebfmi min",min(r[4] for r in res))
    print("p min",p.min(),"bonf",0.001/len(p),"edge frac (ranks 0,19)",((R==0)|(R==NDRAW)).mean().round(3),"expect",round(2/(NDRAW+1),3))
    np.save(f"{sys.argv[4]}/ranks_{case}.npy",R)
