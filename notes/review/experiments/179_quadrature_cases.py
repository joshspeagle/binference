"""Issues #179 and #206: every failure case of the adaptive bin averages, with evaluations."""
import warnings,numpy as np
from scipy.stats import norm
import binference as b
from binference._quadrature import bin_log_average as q
class C:
    def __init__(s,f): s.f=f; s.n=0
    def __call__(s,z,*a):
        s.n+=np.size(z); return s.f(z,*a)
def go(name,f,bins,scale,**k):
    c=C(f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always"); r=q(c,bins,**k)
    print(f"{name:30s}",np.round(np.exp(r)*scale,6).ravel(),"evals",c.n,[str(x.message)[:30] for x in w])
B1=b.Bins.linear(0,1,1);B2=b.Bins.linear(0,2,2)
for s in (1e-2,3e-3,1e-3,1e-4,1e-5,1e-7): go(f"peak s={s}",lambda z:norm.logpdf(z,.5,s)[None],B1,1)
go("2 bins",lambda z:norm.logpdf(z,.5,1e-3)[None],B2,1)
go("spike+flat exact 1",lambda z:np.logaddexp(np.log(.5)+norm.logpdf(z,.5,1e-3),np.log(.5))[None],B1,1)
go("other bin [1,1]",lambda z:np.logaddexp(norm.logpdf(z,.5,1e-3),norm.logpdf(z,1.5,.03))[None],B2,1)
go("10:1 [10,1]",lambda z:np.logaddexp(np.log(10)+norm.logpdf(z,.5,1e-3),norm.logpdf(z,1.5,.03))[None],B2,1)
with np.errstate(divide='ignore'):
    go("underflow s=1e-4 linear",lambda z:np.log(norm.pdf(z,.5,1e-4))[None],B2,1)
    go("underflow via masses",lambda z:np.log(norm.pdf(z,.5,3e-4))[None],B1,1)
go("box",lambda z:np.where(abs(z-.5)<1e-3,np.log(500),-np.inf)[None],B1,1)
go("per row",lambda z,r:norm.logpdf(z,.5,1e-3),B1,1,rows=2)
go("spike+flat rows",lambda z,r:np.logaddexp(np.log(.5)+norm.logpdf(z,.5,1e-3),np.log(.5)),B1,1,rows=2)
b2=b.Bins.grid(np.linspace(0,1,2),np.linspace(0,1,2))
go("2D spike+flat",lambda z:np.logaddexp(np.log(.5)+norm.logpdf(z[...,0],.5,1e-2)+norm.logpdf(z[...,1],.5,1e-2),np.log(.5))[None],b2,1)
