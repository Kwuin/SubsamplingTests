from viabel.models import SubsamplingModel, Model
import jax.numpy as jnp
from jax import random
from viabel.approximations import MFGaussian
from viabel.convenience import newbbvi, bbvi
import stan
import jax
from jax import value_and_grad

seed = 42
key = random.PRNGKey(seed)

def gaussian_log_prior(param):
    return -jnp.sum(param ** 2)

def logistic_function(x):
    return 1 / (1 + jnp.exp(-x))

def logistic_regression_log_likelihood(param, data):
    """
    param: sample_size * dim
    X: data_size * dim
    Y: data_size * 1
    """
    
    dim = data.shape[1] -  1
    X = data[:,:dim]
    Y = data[:, -1]
    #print(param.shape)
    
    param = param[:, jnp.newaxis, :]
    
    logits = logistic_function(jnp.sum(param * X, axis = -1)) # sample_size * data_size 
    #print("MMMMMMMAXXXXXXX")
    @jax.jit
    def my_max(array):
        # some computations...
        max_value = jnp.nanmax(array) 
        return max_value  


    jit = 1e-7
    log_likelihood =  jnp.log(logits + jit) * Y +  jnp.log(1-logits + jit) * (1 - Y)
    
    #print(jnp.isnan(param).any())
    
    if jnp.isnan(log_likelihood).any():
        raise RuntimeError
    return log_likelihood

dim = 5
data_size = 500
X = random.uniform(key ,shape = (data_size, dim))
#true_param = jnp.arange(dim)
true_param = jnp.ones(dim)
prob_x = logistic_function(jnp.dot(X,true_param))
Y = prob_x > 0.5
Y = Y.astype(float)

Y = Y.reshape(data_size, 1)
print(Y)

data = jnp.hstack((X, Y))


#def logistic_regression_log_density(param):
#    return gaussian_log_prior(param) + logistic_regression_log_likelihood(param, data)


#ordinary_model = Model(gaussian_log_density)


res = newbbvi(dim, num_mc_samples = 10, log_prior=gaussian_log_prior, 
                 log_likelihood=logistic_regression_log_likelihood,dataset=data, subsample_size=500, learning_rate=1e-2)


#sm = SubsamplingModel(gaussian_log_prior, logistic_regression_log_likelihood, data, 50)

#array_zeros = jnp.zeros(dim)
#array_twos = jnp.full(dim, 2)
#combined_array = jnp.concatenate((array_zeros, array_twos))


# =============================================================================
# init = jnp.array([ 0.0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
#                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
#                         28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
#                         42, 43, 44, 45, 46, 47, 48, 49]
# )
# init = jnp.atleast_2d(init)
# =============================================================================

# =============================================================================
# def f(param):
#     
#     likelihood = jnp.sum(
#         logistic_regression_log_likelihood(param, data), axis=-1)
#     # print(self.log_likelihood(x, subsample_data))
#     #return jnp.mean(likelihood + gaussian_log_prior(param))
#     return jnp.mean(likelihood)
#     #return jnp.mean( gaussian_log_prior(param))
#     
# =============================================================================

#n_ones = 10 * jnp.atleast_2d(jnp.ones(dim))


#g = jax.value_and_grad(f)
#print(g(n_ones))
print(res["opt_param"])




