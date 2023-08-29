from SpectralDerivative import *


def filter_operations_hsi2(hypercube_data, parameters):
    hypercube_spectral_derivative = SpectralDerivative(hypercube_data,
                                                       parameters["step_length_derivative"]) \
        .symmetrical_first_order_derivative()
    hypercube_spectral_derivative_transpose = hypercube_spectral_derivative.T
    return hypercube_spectral_derivative_transpose
