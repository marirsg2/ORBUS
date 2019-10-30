# Linear Regression to verify implementation
from sklearn import linear_model

import numpy as np

# Scipy for statistics
import scipy

# PyMC3 for Bayesian Inference
import pymc3 as pm

#theano tensors
import theano.tensor as tt

from Evaluation.sanity_check_experiments.MLE_vs_BLM.settings import *

def train_BLM_model(X, y, random_seed = None):
    if random_seed is not None:
        np.random.seed(random_seed)

    with pm.Model() as model:
        # Intercept
        alpha = pm.Normal('alpha', mu = 0, sd = sd_for_priors)

        mu = np.zeros(number_of_dimensions)
        cov = np.diag(np.full(number_of_dimensions, sd_for_priors))
        # Slope
        betas = pm.MvNormal('betas', mu=mu, cov=cov, shape=(number_of_dimensions,))

        # Standard deviation
        sigma = pm.HalfNormal('sigma', sd = sd_for_priors)

        # Estimate of mean
        mean = alpha + tt.dot(X, betas)

        # Observed values
        Y_obs = pm.Normal('Y_obs', mu = mean, sd = sigma, observed = y)

        # Sampler
        step = pm.NUTS()

        # Posterior distribution
        linear_trace = pm.sample(no_of_samples, step, chains=chains)
    return linear_trace




