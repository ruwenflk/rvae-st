import numpy as np


def get_elbo(val, sequence_length, channels, alpha, beta):
    constant = -0.5 * np.log(beta / alpha * np.pi) * sequence_length * channels
    elbo = -val / beta + constant
    return elbo


def get_elbo_reduced(val, sequence_length, channels, alpha, beta):
    elbo = get_elbo(val, sequence_length, channels, alpha, beta)
    return elbo / (sequence_length * channels)
