#!/usr/bin/env python

"""
Channel Identification algorithms that Identifies the Dendritic Processing in a
[Filter]-[Ideal IAF] neural circuit.

- iad_encode     - ideal IAF time encoding machine.
- cim_ideal_iaf  - channel identification machine for ideal IAF neuron.

"""

__all__ = ['cim_ideal_iaf','iaf_encode']

import numpy as np
import scipy as sp
from scipy.signal import fftconvolve

__pinv_rcond__ = 1e-7    # threshold for pseudo inverse computation

def iaf_encode(dt, t, u, b, d, C=1.0):
    """
    IAF time encoding machine.
    
    Encode a finite length signal with an Integrate-and-Fire neuron.
    
    Parameters
    ----------
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is 1/dt Hz.
    t : array_like of floats
        time course of Signal.
    u : array_like of floats
        Signal to encode.
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    C : float
        Neuron capacitance.
    
    Returns
    -------
    s, v : list of floats, ndarray of floats
        returns the signal encoded as an array s of spike times, and the
        voltage trace v of the encoder output.
    
    Notes
    -----
    This is a simplified version of IAF_ENCODE in the ted.python toolbox.
    Interested users may wish to have a look at the Time Encoding and Decoding
    Machine toolbox in the Bionet repository.
    
    """
    s = []                          # initialize the spikes list
    v = np.zeros_like(t)            # initialize the voltage trace
    e = 0;                          # initialize the integration error term

    for i in xrange(1,len(t)):
        # compute the membrane voltage
        v[i] = e + v[i-1] + ( b + 0.5*(u[i]+u[i-1]) )*dt/C
        
        # if the voltage is above the threshold d, generate a spike
        if v[i] >= d:
            s.append( t[i] )        # generate a spike
            v[i] = v[i]-d           # reset the voltage
            e = v[i]                # keep track of the integration error           
        else:
            e = 0
    return s, v

def cim_ideal_iaf(dt, t_Ph, t, u, W, b, d, k, tk, tau, N_max, Calc_MSE_N=False):
    """
    Channel Identification Machine for ideal IAF neuron
    
    Identify linear dendritic processing filter in cascade with a spiking 
    neuron model.
    
    Parameters
    ----------
    dt: float
        Sampling resolution of filter and the input signal.
    t_PH: array_like of floats
        Time course of the filter.
    t: array_like of floats
        Time course of the input signal.
    u: array_like of floats
        Input signal.
    W: floats
        Bandwidth of the input signal.
    b: floats
        Bias of the IAF neuron.
    d: floats
        Firing threshold of the IAF neuron.
    k: floats
        Capacitance of the IAF neuron.
    tk: list of floats
        Spike time sequence
    tau: 1-by-2 array of floats
        Temporal window for identification.
    N_max: integer
        Maximum number of temporal windows used for identification.
    Calc_MSE_N: boolean: boolean
        Identify filter using the first i windows of WINDOWS, for i from 1 to
        N_MAX where N_MAX is the maximum number of windows.
    
    Returns
    -------
    Ph, windows: array_like of floats, list of 1-by-2 arrays
        If `Calc_MSE_N` == False, returns the identified filter projection Ph,
        and the windows used for identification.
    Ph, windows, h_rec_N: array_like of floats, list of 1-by-2 arrays, 
                          list of array of floats array
        If `Calc_MSE_N` == True, returns the identified filter projection Ph,
        and the windows used for identification, as well as the identified
        filter projections using the first i windows of WINDOWS, for i from 1
        to N_MAX where N_MAX is the maximum number of windows.
    
    Notes:
    ------
    The notation used in this code follows from the NIPS'10 paper.
    
    """
    
    # Compute the length of the temporal window
    W_length = dt*round((tau[1]-tau[0])/dt) 
    # Get the first window anchor (the spike based on which we pick the window)
    window_anchor = [x for x in tk if x >= (t[0]+W_length) ]    # find all feasible tk that can be used as window anchors
    window_start  = window_anchor[1] - dt                       # get the start time of the first window
    window_end    = window_start+W_length
    # Initialize temporal window parameters
    windows = []              # cell array of temporal windows
    tk_in_window = []         # cell array of spikes in these window
    all_tk = []               # array of spikes contained in all temporal window

    # While still have enough of the corresponding input signal, keep
    # generating temporal windows.
    while window_end < t[-1]:
        #Generate the temporal window
        windows.append([window_start, window_end])
        
        # Find all spikes falling in the current temporal window
        tk_in_window.append( np.array([x for x in tk if window_start <= x 
                                                        and x <= window_end]) )
        
        # Find the biggest gap in the combined spike train
        all_tk.extend( tk_in_window[-1]-window_start )  # combine spikes
        all_tk.sort();                                  # sort combined spikes
        gaps = np.diff(all_tk);                         # compute gaps
        biggest_gap_idx = np.argmax(gaps)               # find the first index for the biggest gap
        
        # Find the next anchor at least a gap_spike_time away from the end of
        # current temporal window
        gap_spike_time = dt*np.round((all_tk[biggest_gap_idx] + all_tk[biggest_gap_idx+1])/(2*dt))
        window_anchor = [x for x in tk if x > (window_end + gap_spike_time) ]

        # If the next anchor exists (may run out of spikes)
        if window_anchor:  # check empty list with pythonic style using implicit booleanness
            window_start = window_anchor[0] - gap_spike_time;      # get the start time of next temporal window
            window_end = window_start+W_length
        else:
            break
    
    # Find windows that contain fewer than 2 spikes     
    idx = [i for i in xrange(len(tk_in_window)) if len(tk_in_window[i]) >= 2]
    # Set the maximum number of windows to be used
    N = min(len(idx),N_max)
    tk_in_window = [tk_in_window[i] for i in idx[:N] ]
    windows = [windows[i] for i in idx[:N] ]

    # Compute q of each window, and merge all q's to a single vector
    q_N  = np.concatenate( [k*d-b*np.diff(x) for x in tk_in_window] )
    
    # Compute T (Corollary 3 of [1])
    T    = (tau[0]+tau[1])/2
    # Compute all t_k^j-tau+T (Corollary 3 of [1])
    sk_N = np.concatenate([x[:-1]-np.mean(y)+T for x,y in zip(tk_in_window,windows)] )
    # Quantize sk_N by the sampling resolution
    sk_N = dt*np.round( (sk_N-t_Ph[0]) / dt ).astype(int) + t_Ph[0]
    
    # Initialize ensemble matrices G_N
    G_N = np.zeros((len(sk_N),len(sk_N))) 
    
    # Compute the matrix G = [G1; G2; ...; GN] block by block. Each block 
    # Gi = [G1i; G2i; ... Gki] associates with the spikes in the i-th window.
    # Entries of each row G1i are integrals of different shifted version of 
    # u(t) on the same interval [t^i_k t^i_(k+1)]. To compute every entry 
    # efficiently, we calculate the integral of u(t), U(t), one time, and 
    # find the integral of u(t-s) on [a b] by computing U(b-s)-U(a-s). The 
    # above idea can be vectorized for entries of the same row, since they 
    # have the same integral interval.
    
    row = 0                          # initialize row index
    # compute the integral of u, notice that len(int_u) = len(u)-1
    int_u = sp.integrate.cumtrapz(u,t)
    for win in tk_in_window:
        low_idx = _time2idx( win[0]-sk_N-t[0]-dt, dt, 0, len(u)-2 )
        for tks in win[1:]:
            up_idx = _time2idx( tks-sk_N-t[0]-dt, dt, 0, len(u)-2 )
            G_N[row,:] = int_u[up_idx]-int_u[low_idx]
            low_idx = up_idx
            row += 1
    # Calculate ck_N
    ck_N = np.dot(np.linalg.pinv(G_N,__pinv_rcond__),q_N)
    # Recover the filter
    Ph = _syn_sig(dt,t_Ph,ck_N,sk_N,W)
    # If error as a function of the number of windows is requested
    if Calc_MSE_N:
        h_rec_N = []                   # initialize the output
        # get the end index of the population matrix
        G_ix_end = np.cumsum( [len(x)-1 for x in tk_in_window[:-1]] )   
        for num in G_ix_end:
            G = G_N[:num, :num]        # get the matrix for a population of i neurons
            q = q_N[:num]              # get the q of the first i windows
            sk = sk_N[:num]            # get the sk of the first i windows
            # calculate ck of the first i windows
            ck = np.dot(np.linalg.pinv(G,__pinv_rcond__),q)
            # Recover the filter and append the result to the output
            h_rec_N.append( _syn_sig(dt,t_Ph,ck,sk,W) )
            
        h_rec_N.append(Ph)             # the last entry is the recovery for all N windows     
        return Ph, windows, h_rec_N
    else:
        return Ph, windows

def _syn_sig(dt, t_sig, S, ST, Omega, mode='fft'):
    """
    Synthesize band-limited signal from samples and sample times.
    
    Parameters
    ----------
    dt : float
        Sampling resolution of input signal.
    t_sig: array_like of floats 
        time course of the signal.
    S: list of floats
        samples.
    ST: list of floats
        sample time.
    Omega: float
        bandwidth of the signal.
    
    mode: 'fft' or 'sum'
        'fft' for fft-based synthesis.
    
    Returns
    -------
    v : numpy array of floats
        returns synthesized signal.
    
    Notes
    -----
    FFT-based method is faster and more accurate only for small dt.
    
    """
    v = np.zeros_like(t_sig)
    if mode == 'sum':
        for s,st in zip(S,ST):
            v += s*np.sinc( (t_sig-st)*Omega/np.pi )*Omega/np.pi
    elif mode == 'fft':
        t_len = len(t_sig)
        v[ np.round((ST-t_sig[0])/dt).astype(int) ] = S
        t_sinc = dt*np.arange(-t_len+1,t_len+1)
        g = np.sinc(t_sinc*Omega/np.pi)*Omega/np.pi
        v = fftconvolve(g,v)[t_len-1:2*t_len-1]
    return v

def _time2idx( time, dt, lowerBound, upperBound ):
    """
    Convert a continuous-valued array to a discrete integer array. The result
    is hard limited by the given bounds.
    
    Parameters
    ----------
    t : array_like of floats
        time course of Signal.
    dt : float
        Sampling resolution of input signal.
    lowerBound: 
        lower bound of the resultant index.
    upperBound: 
        upper bound of the resultant index.
    
    Returns
    -------
    idx : numpy array of integer
        returns discretized version of TIME.
    
    Notes
    -----
    
    """
    idx = np.round( time/dt ).astype(int)
    idx[ idx < lowerBound]  = lowerBound
    idx[ idx > upperBound]  = upperBound
    return idx