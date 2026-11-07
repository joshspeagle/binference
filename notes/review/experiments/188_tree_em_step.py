"""Issue #188: time of an EM step with a Polya tree, one BLAS thread, on each checkout."""
import time, numpy as np, binference as b
from binference import priors as P
from binference.engines.optimize import em_update
for K in (256,1024):
    bins=b.Bins.linear(0,1,K)
    mock=b.simulate.mock_1d(2000,bins,rng=0)
    m=b.PopulationModel(mock.likelihoods,P.PolyaTree())
    v=np.full(K,1/K); best=1e9
    for _ in range(6):
        t=time.perf_counter()
        for _ in range(50): em_update(m,v)
        best=min(best,(time.perf_counter()-t)/50)
    print(K,"EM step ms",round(best*1e3,3))
    c=np.random.default_rng(0).gamma(1,1,K); bound=m.prior; best=1e9
    for _ in range(6):
        t=time.perf_counter()
        for _ in range(200): bound.map_from_counts(c)
        best=min(best,(time.perf_counter()-t)/200)
    print(K,"map_from_counts ms",round(best*1e3,3))
