#!/usr/bin/env python

"""
Channel Identification algorithms

- cim_ideal_iaf         - CIM with ideal IAF neuron.

"""

__all__ = ['cim_ideal_iaf']

import numpy as np
import scipy as sp
import pdb,time

__pinv_rcond__ = 1e-8
def _synthesize_filter_conv(dt,t_Ph,ck_N,W,tk_in_window,windows,tau):
    Sample = np.zeros_like(t_Ph)
    ck_counter = 0
    for win,tk_in_win in zip(windows,tk_in_window):
        for tk in tk_in_win[:-1]:
            idx = round( ( tk - win[0] + tau -t_Ph[0] )/dt )
            Sample[idx] += ck_N[ck_counter]
            ck_counter += 1
    Ph = sp.signal.fftconvolve(np.sinc(t_Ph*W/np.pi)* W/ np.pi,Sample)
    return Ph[len(t_Ph)/2:-len(t_Ph)/2+1]
def _synthesize_filter(t_Ph,ck_N,W,tk_in_window,windows,tau):
    Ph = np.zeros_like(t_Ph)
    ck_counter = 0
    for win,tk_in_win in zip(windows,tk_in_window):
        for tk in tk_in_win[:-1]:
            Ph += ck_N[ck_counter] * W/ np.pi *\
                    np.sinc( W*(t_Ph - tk + win[0] - tau )/np.pi)
            ck_counter += 1
    return Ph

def cim_ideal_iaf(dt, t_Ph, t, u, W, b, d, k, tk, tau, N_max, Calc_MSE_N=False):
    """
    Channel Identification Machine for ideal IAF neuron
    
    Identify linear dendritic processing filter in cascade with a spiking 
    neuron model.
    
    Parameters
    ----------
    dt: float
        Sampling resolution of filter and the input signal.
    t_PH: array of floats
        Time course of the filter.
    t: array of floats
        Time course of the input signal.
    u: array of floats
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
    tau: 2-by-1 list of floats
        Temporal window.
    N_max: integer
        Maximum number of temporal windows used for identificaiton.
        
    Calc_MSE_N: boolean
        Identify filter using the first i windows of WINDOWS, for i from 1 to
        N_MAX where N_MAX is the maximum number of windows.
    Returns
    -------
    Ph:
    windows:
    Ph_N:
    
    Notes:
    ------
        
    """
    
    # Compute the length of the temporal window
    W_length = dt*round((tau[1]-tau[0])/dt) 
    # Get the first window anchor (the spike based on which we pick the window)
    window_anchor = [x for x in tk if x >= (t[0]+W_length) ]    # find all feasible tk that can be used as window anchors
    window_start  = window_anchor[1] - dt;                      # get the start time of the first window
    window_end    = window_start+W_length
    # Initialize temporal window parameters
    windows = [];             # cell array of temporal windows
    tk_in_window = [];        # cell array of spikes in these window
    all_tk = [];              # array of spikes contained in all temporal window

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
#        pdb.set_trace() 
        # Find the next anchor at least a gap_spike_time away from the end of
        # current temporal window
        gap_spike_time = dt*np.round((all_tk[biggest_gap_idx] + all_tk[biggest_gap_idx+1])/(2*dt));
        window_anchor = [x for x in tk if x > (window_end + gap_spike_time) ]
        #pdb.set_trace()
        # If the next anchor exists (may run out of spikes)
        if window_anchor:  # check empty list with pythonic style using implicit booleanness. 
            window_start = window_anchor[0] - gap_spike_time;      # get the start time of next temporal window
            window_end = window_start+W_length
        else:
            break;
#    pdb.set_trace()    
    # Find windows that contain fewer than 2 spikes     
    idx = [i for i in xrange(len(tk_in_window)) if len(tk_in_window[i]) >= 2]
    # Set the maximum number of windows to be used.
    N = min(len(idx),N_max)
    tk_in_window = [tk_in_window[i] for i in idx[:N] ]
    windows = [windows[i] for i in idx[:N] ]

    # Compute total number of spikes that are contained in temporal windows
    num_tk = sum( [len(x) for x in tk_in_window] )

    # Compute the matrix G_N
    G_N = np.zeros((num_tk-len(windows),num_tk-len(windows)))     # initialize ensemble matrices G_N
    q_N = np.zeros((num_tk-len(windows),1));                          # initialize ensemble matrices q_N

    shifted_u = np.zeros_like(t);
    # build the matrix G_N column by column hence the integral of the shifted
    # stimulus will only be computed once.
    col_idx = 0
    #for i in xrange(N):                 # column block index
    for win,tk_in_win in zip(windows,tk_in_window):
        # get the shift imposed by spikes from the window i
        shift = np.round((tk_in_win - win[0] + tau[0])/dt)
        
        # for every column
        for k_cntr in xrange(len(shift)-1):              
            if shift[k_cntr]>0:
                shifted_u[:shift[k_cntr]] = 0
                shifted_u[shift[k_cntr]:] = u[:-shift[k_cntr]]
            else:
                shifted_u[:shift[k_cntr]] = u[abs(shift[k_cntr]):] 
                shifted_u[shift[k_cntr]:] = 0
            
            # compute the integral of shifted stimulus
            shifted_u_int = dt*sp.integrate.cumtrapz(shifted_u)
            
            # compute G_N, row block by row block
            row_offset = 0;
            for tk_in_rowblk in tk_in_window:
                idx = np.round( ( tk_in_rowblk - t[0] ) / dt ).astype(np.int)
                idx[ idx<0 ] = 0
                idx[ idx>=len(t) ] = len(t)-1
                # Read out the value from the integral of the shifted stimulus
                G_N[ row_offset:row_offset+len(tk_in_rowblk)-1, col_idx ]\
                    = np.diff( shifted_u_int[idx] )
                row_offset += len(tk_in_rowblk)-1;
            
            q_N[col_idx] = k*d - b*(tk_in_win[k_cntr+1]-tk_in_win[k_cntr])
            col_idx += 1
            
    # Calculate ck_N    
    ck_N = np.dot(np.linalg.pinv(G_N,__pinv_rcond__),q_N);
    Ph = _synthesize_filter_conv(dt,t_Ph,ck_N,W,tk_in_window,windows,tau[0])
    
    if Calc_MSE_N:
        h_rec_N = []
        G_ix_end = 0                            # get the end index of the population matrix
        for idx,win in enumerate(tk_in_window[:-1]):
            t0 = time.clock()
            # get the matrix G corresponding to first i windows
            G_ix_end += len(win)-1
            G = G_N[:G_ix_end, :G_ix_end]       # get the matrix for a population of i neurons
            q = q_N[:G_ix_end]                  # get the q_i
            ck_N = np.dot(np.linalg.pinv(G,__pinv_rcond__),q)  # calculate cks
            h_rec_N.append( _synthesize_filter_conv(dt,t_Ph,ck_N,W,tk_in_window[:idx+1],windows[:idx+1],tau[0]) )
            
        h_rec_N.append(Ph)                  # the last entry is the recovery for all N windows     
        return Ph, windows, h_rec_N
    else:
        return Ph, windows
        