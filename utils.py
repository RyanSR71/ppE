import numpy as np
from numpy.linalg import inv
import bilby
import random
import time as tm
import sys
import scipy
from bilby.core import utils
from bilby.core.series import CoupledTimeAndFrequencySeries
from bilby.core.utils import PropertyAccessor
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
import lal
import matplotlib.pyplot as plt


def keplerian_velocity(freqs, total_mass, ppe_b = 1.0):
    # If the total mass is given in solar masses,
    # the frequency should be given in solar masses^{-1}.
    # Likewise for if the total mass is given in seconds.
    return pow(np.pi * total_mass * freqs, ppe_b/3.0)

def ppe_inspiral_correction_to_phase(freqs, total_mass, ppe_beta, ppe_b):
    return ppe_beta * keplerian_velocity(freqs, total_mass * lal.MTSUN_SI, ppe_b)

def ppe_post_inspiral_correction_to_phase(freqs, total_mass, ppe_beta, ppe_b, ppe_delta_epsilon=0.0, Mfreq_IM=0.018):
    velocities_3 = np.pi * total_mass * lal.MTSUN_SI * freqs
    velocity_IM_3 = np.pi * Mfreq_IM
    velocity_IM_b = pow(velocity_IM_3, ppe_b/3.0)
    return ppe_beta * velocity_IM_b + ppe_b / 3.0 * (ppe_beta + ppe_delta_epsilon) * velocity_IM_b * (velocities_3/velocity_IM_3 - 1.0)

def dphi0(phi1, phi2, dphi1, dphi2, f1, f2):
    return (phi2-phi1)/(f2-f1)
def dlnfA(phi1, phi2, dphi1, dphi2, f1, f2):
    return np.log(f2/f1)/(f2-f1)
def dfn3B(phi1, phi2, dphi1, dphi2, f1, f2):
    return -(f2**-3-f1**-3)/(f2-f1)/3.0
def termC(phi1, phi2, dphi1, dphi2, f1, f2):
    return dphi0(phi1, phi2, dphi1, dphi2, f1, f2) - dphi1
def termF(phi1, phi2, dphi1, dphi2, f1, f2):
    return dphi0(phi1, phi2, dphi1, dphi2, f1, f2) - dphi2
def termD(phi1, phi2, dphi1, dphi2, f1, f2):
    return dlnfA(phi1, phi2, dphi1, dphi2, f1, f2) - f1**-1
def termG(phi1, phi2, dphi1, dphi2, f1, f2):
    return dlnfA(phi1, phi2, dphi1, dphi2, f1, f2) - f2**-1
def termE(phi1, phi2, dphi1, dphi2, f1, f2):
    return dfn3B(phi1, phi2, dphi1, dphi2, f1, f2) - f1**-4
def termH(phi1, phi2, dphi1, dphi2, f1, f2):
    return dfn3B(phi1, phi2, dphi1, dphi2, f1, f2) - f2**-4

def ppe_transition_correction_to_phase(freqs, phi_IM, phi_MR, dphi_IM, dphi_MR, f_IM, f_MR):
    C = termC(phi_IM, phi_MR, dphi_IM, dphi_MR, f_IM, f_MR)
    G = termG(phi_IM, phi_MR, dphi_IM, dphi_MR, f_IM, f_MR)
    F = termF(phi_IM, phi_MR, dphi_IM, dphi_MR, f_IM, f_MR)
    D = termD(phi_IM, phi_MR, dphi_IM, dphi_MR, f_IM, f_MR)
    E = termE(phi_IM, phi_MR, dphi_IM, dphi_MR, f_IM, f_MR)
    H = termH(phi_IM, phi_MR, dphi_IM, dphi_MR, f_IM, f_MR)
    one_over_denom = 1.0/(E*G-H*D)
    delta_beta3 = (C*G-F*D) * one_over_denom
    delta_beta2 = (F*E-C*H) * one_over_denom
    delta_beta1 = dphi_IM - delta_beta2 * f_IM ** -1 - delta_beta3 * f_IM ** -4
    delta_beta0 = phi_IM - delta_beta1 * f_IM - delta_beta2 * np.log(f_IM) + delta_beta3 * f_IM ** -3 / 3.0
    return  delta_beta0 + delta_beta1 * freqs + delta_beta2 * np.log(freqs) - delta_beta3 * freqs ** -3 / 3.0

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def ringdown_frequency(strain,frequency_array, total_mass):
    """Numerically finds the ringdown frequency, f_RD, given a frequency
    domain waveform. The ringdown frequency is as defined in:
    https://arxiv.org/pdf/1508.07253.pdf

    Parameters
    ----------
    fd_waveform : FrequencySeries
        The frequency domain waveform of the coalescence.
    total_mass : float
        The total mass of the binary, in solar masses.

    Returns
    -------
    f_RD : float
         The ringdown frequency, in Hz.
    """
    freqs = frequency_array
    i_fLR = int(1.0/(6.0*np.pi * total_mass * lal.MTSUN_SI * (freqs[1]-freqs[0])))
    i_min = np.argmin(np.gradient(np.unwrap(np.angle(strain[i_fLR:2*i_fLR]))))
    f_RD = freqs[i_min + i_fLR]
    return f_RD

def apply_ppe_correction(strain,frequency_array, total_mass, ppe_beta, ppe_b, ppe_delta_epsilon=0.0, Mfreq_IM = 0.018, ratio_f_MR_to_f_RD=0.5, aligned=True, window_size = 41):
    """Applies the parameterized post-Einsteinian corrections to a waveform
    in the frequency domain such that the phase is first-differentiable across
    the different frequency regimes, where a different correction is
    applied in each one. These frequency regimes are:
    * Up to Mf = Mfreq_IM:
    The ppe-beta correction is applied, to the phase through
    delta phi = beta * v ^ b, where v = (pi * M * f) ^ (1 / 3).
    * After Mf = ratio_f_MR_to_f_RD * Mf_RD:
    The ppe-epsilon correction is applied.
    * In between:
    The transition correction is applied.

    Parameters
    ----------
    fd_waveform : FrequencySeries
        The frequency domain waveform of the coalescence.
    total_mass : float
        The total mass of the binary, in solar masses.
    ppe_beta : float
        The ppE parameter beta.
    ppe_b : float
        The ppE parameter b.
    ppe_delta_epsilon :float
        The ppE parameter delta epsilon.
    ratio_f_MR_to_f_RD : float
        The fraction of the ringdown frequency at which
        the intermediate regime ends and becomes the merger-
        ringdown regime.
        
    Returns
    -------
    new_fd_waveform : FrequencySeries
         The frequency domain waveform of the coalescence, after applying
         the ppE corrections.
    """
    total_mass_in_seconds = total_mass * lal.MTSUN_SI
    new_fd_waveform = strain
    
    # Array of frequencies, in Hz.
    freqs = frequency_array

    # Transition frequency indices.
    i_IM = int(round(0.018 / (total_mass_in_seconds * (freqs[1]-freqs[0]))))
    i_MR = int(round(ratio_f_MR_to_f_RD * ringdown_frequency(strain,frequency_array, total_mass) / (freqs[1]-freqs[0])))

    # Transition frequencies, in Hz.
    f_IM = freqs[i_IM]
    f_MR = freqs[i_MR]

    # Transition velocities, dimensionless.
    v_IM = keplerian_velocity(f_IM, total_mass_in_seconds)

    # Array of phase changes.
    phase_change = np.zeros_like(freqs)

    #  Compute ppe-beta correction:
    phase_change[1:i_IM+1] = ppe_inspiral_correction_to_phase(freqs[1:i_IM+1], total_mass, ppe_beta, ppe_b)

    # Compute ppe-epsilon correction:
    phase_change[i_MR:] = ppe_post_inspiral_correction_to_phase(freqs[i_MR:], total_mass, ppe_beta, ppe_b, ppe_delta_epsilon, Mfreq_IM=total_mass_in_seconds*f_IM)
    
    # Compute transition correction, if any:
    ppe_epsilon = ppe_beta + ppe_delta_epsilon
    dphi_factor = ppe_b * np.pi * total_mass_in_seconds / 3.0 * pow(v_IM,ppe_b-3)
    phase_change[i_IM:i_MR] = ppe_transition_correction_to_phase(freqs[i_IM:i_MR], phase_change[i_IM], phase_change[i_MR], dphi_factor * ppe_beta, dphi_factor * ppe_epsilon, f_IM, f_MR)

    # Apply Savitsky-Golay Filter to phase_change to reduce sharp transitions:
    # phase_change = savitzky_golay(phase_change, window_size,1)
    
    # The time and phase shift incurred.
    phase_change_to_align = ppe_beta * pow(v_IM, ppe_b) + ppe_b / 3.0 * ppe_epsilon * pow(v_IM, ppe_b) * (freqs/f_IM - 1.0)

    if aligned == False:
        new_fd_waveform = new_fd_waveform * np.exp(1j * phase_change)
    
    if aligned == True:
        new_fd_waveform = new_fd_waveform * np.exp(1j * (phase_change - phase_change_to_align))

    return new_fd_waveform



def integral_Ib(p_0, exponent_b):
    b = exponent_b
    if b != 4:
        return 4.0 * (pow(p_0, b) - p_0 ** 4) / (4.0 - b)
    else: # b == 4:
        return -4.0 * p_0 ** 4 * np.log(p_0)
    
def integral_Ib_plus(p_0, p_1, exponent_b):
    b = exponent_b
    if b != 4:
        return 4.0 * p_0 * (1 - pow(p_1, b-4)) / (4.0 - b)
    else: # b == 4:
        return 4.0 * p_0 ** 4 * np.log(p_1)
    
def integral_I2bb(p_0, exponent_b1, exponent_b2):
    return integral_Ib(p_0, exponent_b1 + exponent_b2) - \
            integral_Ib(p_0, exponent_b1) - \
            integral_Ib(p_0, exponent_b2) + \
            integral_Ib(p_0, 0.0)

# Used for debugging
def Phi_b_C0(p_0, p_1, exponent_b):
    one_over_denom = 1.0 / (1 - (p_0/p_1)**4)
    return one_over_denom * \
           integral_I2bb(p_0, exponent_b, exponent_b)

# Used for debugging
def Phi_b_C1(p_0, p_1, exponent_b):
    one_over_denom = 1.0 / (1 - (p_0/p_1)**4)
    return one_over_denom * \
           (integral_I2bb(p_0, exponent_b, exponent_b) - \
           2 * exponent_b / 3.0 * integral_I2bb(p_0, exponent_b, 3) + \
           exponent_b ** 2 / 9.0 * integral_I2bb(p_0, 3, 3))

# Used for debugging
def match_floor_approximant_C0(p_0, p_1, exponent_b, beta, u_IM):
    return 1.0 - 0.5 * beta ** 2 * pow(u_IM, 2.0 * b) * Phi_b_C0(p_0, p_1, exponent_b)

# Used for debugging
def match_floor_approximant_C1(p_0, p_1, exponent_b, beta, u_IM):
    return 1.0 - 0.5 * beta ** 2 * pow(u_IM, 2.0 * b) * Phi_b_C1(p_0, p_1, exponent_b)


def matrix_P(p_0, p_1):
    two_over_denom = 2.0 / (1 - (p_0/p_1)**4)
    P11 = integral_Ib(p_0, 6) + integral_Ib_plus(p_0, p_1, 6)
    P12 = -integral_Ib(p_0, 3) - integral_Ib_plus(p_0, p_1, 3)
    P21 = P12
    P22 = integral_Ib(p_0, 0) + integral_Ib_plus(p_0, p_1, 0)
    return two_over_denom * np.array([[P11,P12],[P21,P22]])

def vector_p_b(p_0, p_1, exponent_b):
    two_over_denom = 2.0 / (1 - (p_0/p_1)**4)
    p1 = -integral_Ib(p_0, exponent_b + 3) + integral_Ib(p_0, 3) + \
          exponent_b / 3.0 * (integral_Ib(p_0, 6) - integral_Ib(p_0, 3))
    p2 = integral_Ib(p_0, exponent_b) - integral_Ib(p_0, 0) - \
          exponent_b / 3.0 * (integral_Ib(p_0, 3) - integral_Ib(p_0, 0))
    return two_over_denom * np.array([p1, p2])
    
def term_under_sqrt(p_0, p_1, exponent_b):
    vec_p_b = vector_p_b(p_0, p_1, exponent_b)
    vec_theta_0 = -np.dot(inv(matrix_P(p_0, p_1)), vec_p_b)
    return Phi_b_C1(p_0, p_1, exponent_b) + 0.5 * np.dot(vec_theta_0, vec_p_b)


def beta_from_beta_tilde_paper(beta_tilde, p_0, p_1, exponent_b, u_IM):
    """Produces ppE beta from the given parameters. This formulation is in
    terms of the parameters used in the paper.

    Parameters
    ----------
    beta_tilde : float
        The rescaled beta term (which can be computed from a desired mismatch).
    p_0 : float
        The lower frequency cutoff of the waveform, in units of fractional
        inspiral velocity: p_0 = (f_0 / f_IM)^(1/3)
    p_1 : float
        The upper frequency cutoff of the waveform, in units of fractional
        inspiral velocity: p_1 = (f_1 / f_IM)^(1/3)
    exponent_b : float
        The ppE exponent b.
    u_IM: float
        The Keplerian velocity at which the inspiral regime ends.

    Returns
    -------
    beta : float
         The ppE parameter beta.
    """
    denominator = pow(u_IM, exponent_b) * np.sqrt(term_under_sqrt(p_0, p_1, exponent_b))
    return beta_tilde / denominator

def beta_from_beta_tilde_wrapped(beta_tilde, f_min_Hz, Mf_max, exponent_b, Mf_IM, total_mass):
    """Produces ppE beta from the given parameters. This formulation differs
    from that given in the paper.

    Parameters
    ----------
    beta_tilde : float
        The rescaled beta term (which can be computed from a desired mismatch).
    f_min_Hz : float
        The lower frequency cutoff of the waveform, in units of Hz.
    Mf_max : float
        The upper frequency cutoff of the waveform, in dimensionless units.
    exponent_b : float
        The ppE exponent b.
    Mf_IM: float
        The dimensionless frequency at which the inspiral regime ends.
    total_mass: float
        The total mass of the binary in units of solar masses.

    Returns
    -------
    beta : float
         The ppE parameter beta.
    """
    # Units of solar mass can be converting to units of time using the conversion
    # factor: 1 M_solar = 4.925491025543576e-06s.
    # Then 1 M_solar^-1 = 203025.44351700248Hz
    # To convert Hz to inverse solar masses we divide by 203025.44351700248.
    Mf_min = total_mass * f_min_Hz * 4.925491025543576e-06
    p_0 = pow(Mf_min / Mf_IM, 1.0/3.0)
    p_1 = pow(Mf_max / Mf_IM, 1.0/3.0)
    u_IM = pow(np.pi * Mf_IM, 1.0/3.0)
    return beta_from_beta_tilde_paper(beta_tilde, p_0, p_1, exponent_b, u_IM)

## Only used for debugging
def beta_from_beta_tilde_floor_C0(beta_tilde, p_0, p_1, exponent_b, u_IM):
    denominator = pow(u_IM, exponent_b) * np.sqrt(Phi_b_C0(p_0, p_1, exponent_b))
    return beta_tilde / denominator

## Only used for debugging
def beta_from_beta_tilde_floor_C1(beta_tilde, p_0, p_1, exponent_b, u_IM):
    denominator = pow(u_IM, exponent_b) * np.sqrt(Phi_b_C1(p_0, p_1, exponent_b))
    return beta_tilde / denominator