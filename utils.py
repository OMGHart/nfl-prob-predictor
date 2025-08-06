import numpy as np
from scipy.special import logit, expit

def logit_func(y):
    tol = 1e-5
    return logit(np.clip(y, tol, 1-tol))
def expit_func(y):
    return expit(y)