import numpy as np
import scipy.signal as sig

def lowpass1(f_cut, T):
    ''' Lowpass 1nd order butterwort filter synthesis.
    '''
    a = np.exp(-2*np.pi*f_cut*T)
    return [1-a, 0, 0], [1, -a, 0]
    
def lowpass2(f_cut, T):
    ''' Lowpass 2nd order butterwort filter synthesis.
    '''
    f_nyquist = 0.5/T
    b, a = sig.butter(2, f_cut/f_nyquist)
    return b, a

def notch2(f_nominal, f_width, T):
    ''' Two parameters Notch (band-stop) filter synthesis.

        Calculate coffecients of notch filter with stop-band center at f_nominal (Hz) with width f_width (Hz).
        T is discretization period (s). Returns numerator and denominatror (b, a) of transfer function. 
    '''
    wn = 2.0*np.pi * f_nominal * T
    wc = 2.0*np.pi * f_width * T
    a = 1.0 / np.cos(wc / 2.0) - np.tan(wc / 2.0)
    K = (1.0 - 2.0*a*np.cos(wn) + a**2) / (2.0 - 2.0*np.cos(wn))
    b = [K, -2.0*K*np.cos(wn), K]
    a = [1.0, -2.0*a*np.cos(wn), a**2]
    return b, a

def notch3(f_nominal, f_width, L_stop, T):
    ''' Notch (band-stop) filter synthesis.

        Calculate coffecients of notch filter with stop-band center at f_nominal (Hz) with width f_width (Hz).
        L_stop (dB) is filter attenuation at f_nominal frequency. It must be lower then zero.
        T is discretization period (s). Returns numerator and denominatror (b, a) of transfer function. 
    '''
    # check parameters
    if T <= 0.0:
        raise ValueError('T must be posititve')
    f_nyquist = 1/(2*T)
    if f_nominal < 0.0 or f_nominal > f_nyquist:
        raise ValueError('f_nominal must be in interval [0.0, 1/(2*T)]')
    if f_width < 0.0 or f_width >= 2*f_nominal:
        raise ValueError('f_width must be in interval [0.0, 2*f_nominal]')
    if L_stop >= 0.0:
        raise ValueError('L_stop must be negative.')
    # convert Hz to rad/s
    Wn = 2*np.pi*f_nominal
    Wc = 2*np.pi*f_width
    # notch parameters
    b = 10**(L_stop/20)
    p = (Wn - Wc/2)/Wn
    a = 0.5 * np.sqrt( (p - 1/p)**2 / (1 - 2*b**2) )
    # tf numerator and denominator
    den = np.array([1, - 2*np.exp(-a*Wn*T)*np.cos(Wn*T*np.sqrt(1-a**2)), np.exp(-2*a*Wn*T)])
    num = np.array([1, - 2*np.exp(-a*b*Wn*T)*np.cos(Wn*T*np.sqrt(1-a**2*b**2)), np.exp(-2*a*b*Wn*T)])
    num *= np.sum(den) / np.sum(num)
    return num, den