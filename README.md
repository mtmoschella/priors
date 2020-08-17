# priors
This is some code I hacked for implementing priors in emcee and pymultinest

The basic use is as follows

```python
myprior = FlatPrior(0.0, 1.0) # creates a Prior object
xmin, xmax = myprior.support()
lnp = myprior(0.5) # evaluates ln (natural log) of the prior function at the requested point
```
