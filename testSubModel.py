
from viabel.models import SubsamplingModel, Model
import jax.numpy as jnp
from jax import random
from viabel.approximations import MFGaussian
from viabel.convenience import newbbvi, bbvi
import stan
import jax

seed = 42
key = random.PRNGKey(seed)

def gaussian_log_prior(param):
    return - jnp.sum(param ** 2)

def gaussian_log_likelihood(param, x):
    
    """
    param:
    """
    param = param[:, jnp.newaxis, :]
    return - jnp.sum((x - param) ** 2, axis=2)

dim = 50
data_size = 5000
true_mean = jnp.arange(dim)

z = random.normal(key, shape=(data_size, dim))
data = true_mean + z

def gaussian_log_density(param):
    return gaussian_log_prior(param) + gaussian_log_likelihood(param, data)

sm = SubsamplingModel(gaussian_log_prior, gaussian_log_likelihood, data, 5, 1)

ordinary_model = Model(gaussian_log_density)

#res = newbbvi(dim,  log_prior=gaussian_log_prior, 
#                 log_likelihood=gaussian_log_likelihood,dataset=data, subsample_size=50)

f = jax.grad(sm)
a = jnp.arange(dim) + 0.1
b = jnp.arange(dim) + 0.2
param = jnp.vstack([a,b])
print(jnp.shape(param))
print(f(param))