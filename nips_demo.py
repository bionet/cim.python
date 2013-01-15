#!/usr/bin/env python

"""
Identifying Dendritic Processing in a [Filter]-[Ideal IAF] neural circuit
This demo illustrates identification of the [Filter] in the
[Filter]-[Ideal IAF] circuit using band-limited input signals, i.e.,
signals that belong to the Paley-Wiener space.

The code below corresponds to Corollary 3 in [1] and was used to generate
Figs. 4-6 in [1]. The employed filter was taken from [2].

Author:               Yevgeniy B. Slutskiy <ys2146@columbia.edu>

Revision Author:      Chung-Heng Yeh <chyeh@ee.columbia.edu>

Bionet Group, Columbia University

Copyright 2010-2012   Yevgeniy B. Slutskiy and Chung-Heng Yeh
"""

import cim

import matplotlib as mpl
#mpl.use('TKAGG')
import matplotlib.pylab as p

from scipy.signal import fftconvolve
import numpy as np
from math import sqrt, log10, factorial
from time import time

def run_time(msg,timer):
    """
    Display the time between the function call and TIMER in "min' sec'" format.
    """
    t = time()-timer
    print msg + " %d\' %.2f\"" % (int(t/60.), t-60.*int(t/60.) )

def low_pass_filter(dt,t_sig,sig,Omega):
    """
    Pass input through a low-pass-filter of bandwidth Omega.
    """
    t_len = len(t_sig)
    t_sinc = dt*np.arange(-t_len+1,t_len+1)
    g = np.sinc(t_sinc*Omega/np.pi)*Omega/np.pi
    return dt*fftconvolve(g,sig)[t_len-1:2*t_len-1]
    
def create_random_band_signal(dt, t_sig, Omega, seed=1):
    """
    Create a random band-limited signal.
    
    Parameters
    ----------
    dt : float
        Sampling resolution of input signal.
    t_sig : array_like of floats
        time course of Signal.
    Omega : float
        bandwidth of the signal.
    seed : integer
        seed for random number generator.
    
    Returns
    -------
    v : ndarray of floats
        returns the synthesized signal.
    
    Notes
    -----
    The idea is to first randomly generate samples of signal, and then pass the
    samples through a low pass filter (LPF). For readers who are not familiar 
    with the above concept, please refer to Shannon Sampling Theorem.
    
    """
    np.random.seed(seed)             # set the seed of random number generator
    v = np.zeros_like(t_sig)         # initialize the signal
    # generate the random samples
    gap = int( (np.pi/Omega)/dt )    
    v[ np.arange(0,len(v),gap) ] = np.random.randn( len(v)/gap )
    # compute the low-pass-filter; the length of LPF is twice the length of
    # the signal in order to cover the entire signal during convolution.
    t_len = len(t_sig)                
    t_sinc = dt*np.arange(-t_len+1,t_len+1)     # set time course of LPF
    g = np.sinc(t_sinc*Omega/np.pi)*Omega/np.pi # compute the LPF
    # convolve samples with LPF and extract the proper portion
    v = fftconvolve(g,v)[t_len-1:2*t_len-1]     
    v /= np.max(np.abs(v))                      # normalize the signal
    return v

def plot_cim_result(dt,t_sig,sig,filt_sig,windows,spk,t_h,h,Ph,h_hat,freq,
                    tau,fig_name='',fig_title='',fig_size=(4,3)):
    """
    Plot the CIM result.
    """
    fig = p.figure(num=fig_name,figsize=fig_size,dpi=300, facecolor='w')
    fig.suptitle(fig_title,fontsize=24)
    # plot the input stimulus u
    ax = fig.add_subplot(3,2,1,xlim=(0,t_sig[-1]-t_sig[0]),ylim=(-1,1),
             ylabel='Amplitude', xlabel='Time, [s]',
             title='$(a)\qquad$Input signal u(t)');
    ax.plot(t_sig - t_sig[0], sig,
            label='$\Omega = 2\pi\cdot '+ repr(int(freq)) + '$rad/s$\qquad$')
    p.legend(loc='lower right')
    #plot the periodogram power spectrum estimate of u
    ax = fig.add_subplot(3,2,2,
            xlim=(-150,150),ylim=(-140,0),
             title='$(d)\qquad$Periodogram Power Spectrum Estimate of u(t)');
    ax.psd(sig,pad_to=len(t_sig),NFFT=len(t_sig),Fs=1/dt,sides='twosided',
           label='supp$(\mathcal{F}u)=[-\Omega,\Omega]\qquad$')
    p.legend(loc='lower right')
    # plot h, Ph and h_hat (the filter identified by the algorithm)
    ax = fig.add_subplot(3,2,3,
             xlim=(0,t_sig[-1]-t_sig[0]),ylim=(0,1.2), xlabel='Time, [s]',
             yticks=[],yticklabels=[],
             title='$(b)\qquad$Output of the [Filter]-[Ideal IAF] neural circuit');
    colorset = mpl.cm.hsv(np.arange(len(windows))/float(len(windows)),1)
    ax.stem(spk-t_sig[0], np.ones_like(spk),linefmt='k-',
            markerfmt='k^',label='$D = 40$Hz')
    for color,win in zip(colorset,windows):
        ax.axvspan(win[0]-t_sig[0],win[1]-t_sig[0],facecolor=color,
                   edgecolor='none',label='_nolegend_')
    p.legend(loc='lower right')
    # plot the periodogram power spectrum estimate of h
    ax = fig.add_subplot(3,2,4,
             xlim=(-150,150),ylim=(-140,0),#ylabel='Power, [dB]',
             title='$(e)\qquad$Periodogram Power Spectrum Estimate of h(t)');
    ax.psd(h,pad_to=len(t_sig),NFFT=len(t_sig),Fs=1/dt,sides='twosided',
           label='supp$(\mathcal{F}h)\supset[-\Omega,\Omega]\qquad$')
    p.legend(loc='lower right')
    # plot the periodogram power spectrum estimate of v=u*h
    ax = fig.add_subplot(3,2,5,xlim=(t_h[0],t_h[-1]), ylim=(-50,100),
             ylabel='Amplitude', xlabel='Time, [s]',
             title='$(c)\qquad$Original filter vs. the identified filter')
    idx = np.logical_and( t_h >= tau[0], t_h <= tau[1] )
    # Normalized RMSE between h and h_hat
    h_hhat_err   = np.abs(h-h_hat)/max(abs(h));               # compute the absolute error 
    h_hhat_RMSE  = sqrt(dt*np.trapz(h_hhat_err[idx]**2)/(tau[1]-tau[0]))     # compute the RMSE
                   
    # Normalized RMSE between Ph and h_hat
    Ph_hhat_err  = np.abs(Ph-h_hat)/max(abs(Ph));                       # compute the absolute error
    Ph_hhat_RMSE = sqrt(dt*np.trapz(Ph_hhat_err[idx]**2)/(tau[1]-tau[0]))    # compute the RMSE

    ax.plot(t_h, h,'--k',label='$h,\,$RMSE$(\hat{h},h)$ = %3.2e $\qquad$' % h_hhat_RMSE )
    ax.plot(t_h, Ph,'-b',label='$\mathcal{P}h,\,$RMSE$(\hat{h},\mathcal{P}h)$ = %3.2e $\qquad$' % Ph_hhat_RMSE)
    ax.plot(t_h, h_hat,'-r',label='$\hat{h}$')
    p.legend(loc='upper right')
    
    # plot the periodogram power spectrum estmate of v=u*h
    ax = fig.add_subplot(3,2,6,
             xlim=(-150,150),ylim=(-140,0),#ylabel='Power, [dB]',
             title='$(f)\qquad$Periodogram Power Spectrum Estimate of v(t)');
    ax.psd(filt_sig,pad_to=len(t_sig),NFFT=len(t_sig),Fs=1/dt,sides='twosided',
           label='supp$(\mathcal{F}v)=[-\Omega,\Omega]\qquad$')
    p.legend(loc='lower right')
    fig.text(.1,0.05,'[1] $Identifying\ Dendritic\ Processing$, ' +
             'Advances in Neural Information Processing Systems 23, pp. 1261-1269, 2010')
    if fig_name:
        p.savefig(fig_name+'.png',dpi=300)
    p.show()
    
if __name__=='__main__':
    # Initialize the demo
    tic_demo = time()  # initialize the timer for the entire demo
    tic_init = time()  # initialize the timer for the initialization
    dt = 5e-6          # set the time step, [s]
    
    # Specify the filter h to be used
    # -------------------------------
    # Generate a filter according to Adelson and Bergen in [2]. h has a
    # temporal support on the interval [T_1, T_2]
    T_1, T_2 = 0., 0.1+dt               # set the interval of the impulse response, [s]
    t_filt = np.arange(T_1,T_2,dt)      # set the length of the impulse response, [s]
    a = 200;                            # set the filter parameter
    h = 3*a*np.exp(-a*t_filt)*((a*t_filt)**3/factorial(3)\
        -(a*t_filt)**5/factorial(5))
    # set zero-padded version of h, for comparison purpose
    h_long = np.concatenate( (np.zeros(len(h)//2),h,np.zeros(len(h)//2)) )
    t_long = np.concatenate(( dt*np.arange(-(len(h)//2),0),t_filt,
                              dt*np.arange(len(h),len(h)+len(h)//2)))
    # Plot the filter
    p.figure(0);
    p.axes(xlim=(t_filt[0],t_filt[-1]),xlabel='Time, [s]',ylabel='Amplitude',
           title='Impulse response $h(t)$ of the filter');
    p.plot(t_filt, h);
    p.show()
    
    # Create a band-limited stimulus. The bandwidth W = 2*pi*25 rad/s
    # --------------------------------------------------------------
    f = 25.                       # set the input signal bandwidth, [Hz]
    W = 2.*np.pi*f                # calculate the bandwidth in [rad]
    t = np.arange(0.,1.12+dt,dt)  # set the time course of the input signal
    # fix the state of random number generator for reproducible results
    u = create_random_band_signal(dt, t, W, seed=20130101)
    # Plot the stimulus
    p.figure(1)
    p.axes(xlim=(t[0],t[-1]),xlabel='Time, [s]',ylabel='Amplitude',
           title='Input Stimulus $u(t)$');  
    p.plot(t,u)
    p.show()

    # Compute the filter projection Ph
    # --------------------------------
    # Ph is the projection of h onto the input signal space. It is the best
    # approximation of h that can be recovered.
    t_Ph = t_long                           # set the time course of the projection Ph
    Ph = low_pass_filter(dt,t_Ph,h_long,W)  # find the projection Ph
    # Plot the filter projection Ph
    p.figure(2)
    p.axes(xlim=(t_Ph[0],t_Ph[-1]),xlabel='Time, [s]',ylabel='Amplitude',
           title='Filter $h(t)$ and Filter projection $\mathcal{P}h(t)$');  
    p.plot(t_Ph, h_long,'--k',label='$h\qquad$' )
    p.plot(t_Ph, Ph,'-b',label='$\mathcal{P}h\qquad$')
    p.legend(loc='upper right')
    p.show()
    
    # Filter the stimulus u
    # -------------------------
    # Since all signals are finite, the filter output v=u*h is not calculated
    # properly on the boundaries. v_proper is that part of v, for which the
    # convolution of u and h is computed correctly.
    v_proper = dt*fftconvolve(u,h,'valid')                        # convolve u with h
    u_proper = u[len(h)-1:]                # get the proper part of v
    t_proper = t[len(h)-1:]                # get the corresponding time vector
    # Plot the filter output
    p.figure(3)
    p.axes(xlim=(0,t_proper[-1]-t_proper[0]),xlabel='Time, [s]',ylabel='Amplitude',
           title='Filter output $v(t)$')
    p.plot(t_proper-t_proper[0],v_proper)
    p.show()
    
    # Encode the filter output v=u*h with an IAF neuron
    # -------------------------------------------------
    # Specify parameters of the Ideal IAF neuron
    delta = 0.007                              # set the threshold
    bias  = 0.28                               # set the bias
    kappa = 1.                                 # set capacitance
    # # Encode the filter output
    spk_train, vol_trace = cim.iaf_encode(dt, t_proper, v_proper, b=bias, d=delta, C=kappa)
    run_time('Initialization time:',tic_init)  # display the initialization time 
    
    # Plot the voltage trace and the associated spike train
    p.figure(4)
    ax1 = p.subplot2grid((7, 1), (0, 0),rowspan=5,xlim=(0., 1.0),
            ylabel='Amplitude', ylim=(min(vol_trace), delta*1.1),
            title='Output of the [Filter]-[Ideal IAF] neural circuit')
    p.setp(ax1.get_xticklabels(), visible=False)
    ax2 = p.subplot2grid((7, 1), (5, 0),rowspan=2,sharex=ax1,
            xlabel='Time, [s]', ylim=(0., 1.1),xlim=(0., 1.0),
            yticks = (),yticklabels = ())
    ax1.plot(t_proper-t_proper[0],vol_trace,'b-',
             label='Membrane voltage $v\qquad$')
    ax1.plot((0, t_proper[-1]-t_proper[0]),(delta,delta),'--r',
             label='Threshold $\delta=' + repr(delta) + '$')
    ax1.plot(spk_train-t_proper[0],delta*np.ones_like(spk_train),'ro',
             label= '$v(t)=\delta$')
    ax1.legend(loc='lower right')
    ax2.stem(spk_train-t_proper[0],np.ones_like(spk_train),
             linefmt='k-', markerfmt='k^', label='$(t_k)_{k\in Z}$')
    ax2.legend(loc='lower right')
    p.show()

    # Identify the filter projection Ph
    # ---------------------------------
    # Since the temporal support of the filter h is not known, we identify the
    # projection Ph in a window tau. Temporal windows W^i of spikes can be 
    # chosen arbitrarily. Here we pick W^i so that the
    # Nyquist-type condition on the density of spikes is achieved quickly (see
    # also Theorem 1 and Remark 1 in [1]).
    T_filt_rec = 0.12;                          # specify the hypothesized length of the impulse response 
    tau = [-(T_filt_rec - t_filt[-1])/2, 0]     # tau is centered around the actual temporal support T_2-T_1
    tau[1] = t_filt[-1]-tau[0];
    
    # set the maximum number of windows to be used (could be smaller depending on the simulation)
    N_max = 10 
    
    # start the algorithm timer
    tic = time()   
    
    # execute the CIM algorithm
    [h_hat, windows] = cim.cim_ideal_iaf(dt, t_Ph, t_proper, u_proper,
                           W, bias, delta, kappa, spk_train, tau, N_max)
    
    # display execution time
    run_time('CIM running time for Fig.4:',tic)
    
    # Generate Fig. 4 of [1]
    # ----------------------
    plot_cim_result(dt,t_proper,u_proper,v_proper,windows,spk_train,
        t_Ph,h_long,Ph,h_hat,f,tau,fig_size = (20,15),fig_name = 'nips_fig_4',
        fig_title='NIPS 2010 Figure 4\nA.A. Lazar and Y.B. Slutskiy')
    
    
    # Generate Fig. 5 of [1]
    # ----------------------
    # The following procedures are same as above except that the input stimulus 
    # is now band-limited to 100Hz.
    
    f = 100.                                    # set the input signal bandwidth, [Hz]
    W = 2.*np.pi*f                              # calculate the bandwidth in [rad]
    t = np.arange(0.,1.52+dt,dt)                # set the time course of the input signal
    # fix the state of random number generator for reproducible results
    u = create_random_band_signal(dt, t, W, seed=20130101)
    
    # Compute the filter projection Ph
    Ph = low_pass_filter(dt,t_Ph,h_long,W)
     
    # Filter the input signal u
    v_proper = dt*fftconvolve(u,h,'valid')  # convolve u with h
    u_proper = u[len(h)-1:]                 # get the proper part of v
    t_proper = t[len(h)-1:]                 # get the corresponding time vector
    
    # Specify parameters of the Ideal IAF neuron
    delta = 0.007                           # set the threshold
    bias  = 0.28                            # set the bias
    kappa = 1.                              # set capacitance
    
    # Encode the filter output
    spk_train, vol_trace = cim.iaf_encode(dt, t_proper, v_proper, b=bias, d=delta, C=kappa)
    
    N_max = 15                              # set the maximum number of windows to be used
    tic = time()                            # start the algorithm timer
    [h_hat, windows] = cim.cim_ideal_iaf(dt, t_Ph, t_proper, u_proper,
                           W, bias, delta, kappa, spk_train,tau, N_max)
    # display execution time
    run_time('CIM running time for Fig.5:',tic) 
    
    # Plot the results
    plot_cim_result(dt,t_proper,u_proper,v_proper,windows,spk_train,
        t_Ph,h_long,Ph,h_hat,f,tau,fig_size = (20,15),fig_name = 'nips_fig_5',
        fig_title='NIPS 2010 Figure 5\nA.A. Lazar and Y.B. Slutskiy')
    
    # Generate Fig. 6 of [1]
    # ----------------------
    # In Fig. 6a we plot the mean square error (MSE) between the filter
    # projection Ph and the identified filter h_hat as a function of the number
    # of temporal windows N.
    #
    # In Fig. 6b we plot the mean square error (MSE) between the original
    # filter h and the identified filter h_hat as a function of the input
    # signal bandwidth
    tic_fig6  = time()                      # initialize the timer for Fig.6
    tic_fig6a = time()                      # initialize the timer for Fig.6a
    f = 100.                                # set the input signal bandwidth, [Hz]
    W = 2.*np.pi*f                          # calculate the bandwidth in [rad]
    t = np.arange(0.,8.5+dt,dt)             # set the time course of the input signal
    # fix the state of random number generator for reproducible results
    u = create_random_band_signal(dt, t, W, seed=20130101)
    
    # Compute the filter projection Ph
    Ph = low_pass_filter(dt,t_Ph,h_long,W)
     
    # Filter the input signal u
    v_proper = dt*fftconvolve(u,h,'valid')  # convolve u with h
    u_proper = u[len(h)-1:]                 # get the proper part of v
    t_proper = t[len(h)-1:]                 # get the corresponding time vector
    
    # Encode the filter output v=u*h with an IAF neuron
    delta = 0.007                           # set the threshold
    kappa = 1.                              # set capacitance
    
    
    D = [20., 40., 60.]                     # set the average spiking density
    N_max = 30                              # set the maximum number of windows to be used
    MSE_N = np.zeros((len(D),N_max))        # initialize the MSE array
    
    idx = np.logical_and( t_Ph >= tau[0], t_Ph <= tau[1] )    # find indices of t for the given frame

    for i,d in enumerate(D):
        bias  = d*delta/kappa;            # set the bias
        # Encode the filter output v=u*h with an IAF neuron
        spk_train, vol =  cim.iaf_encode(dt, t_proper, v_proper, b=bias, d=delta, C=kappa)
        # Identify the filter projection Ph
        [h_hat, windows, h_hat_N] = cim.cim_ideal_iaf(dt, t_Ph, t_proper, u_proper,
                                     W, bias, delta, kappa, spk_train, tau, N_max,
                                     Calc_MSE_N=True)
        # Compute the normalized MSE
        for j in xrange(len(windows)):
            # Normalized RMSE error for Ph - h_hat
            Ph_hhat_err  = abs(Ph-h_hat_N[j])/max(abs(Ph))
            Ph_hhat_RMSE = sqrt(dt*np.trapz(Ph_hhat_err[idx]**2)/(tau[1]-tau[0]))
            MSE_N[i,j] = 10*log10( Ph_hhat_RMSE**2 )
    
    run_time('Fig. 6a time: ', tic_fig6a)  # display the running time for Fig.6a
    
    # Get data for Fig. 6b
    # --------------------
    # In the following, for fixed number of windows N, we compute the MSE
    # between the original filter h and the identified filter h_hat as a
    # function of the input signal bandwidth
    tic_fig6b = time()                         # initialize the timer for Fig.6b
    
    F = np.arange(10.,160.,10.)                # specify the bandwidth vector
    t = np.arange(0,1.8+dt,dt)                 # specify the bandwidth vector
    N_max = 15                                 # set the number of windows
    
    # specify Ideal IAF neuron parameters
    delta = 0.007                              # set the threshold
    bias  = 0.42                               # set the bias
    kappa = 1.                                 # set the capacitance
    
    MSE_BW = np.zeros_like(F)
    for i,f in enumerate(F):
        W = 2*np.pi*f                          # calculate the bandwidth in [rad]
        u = create_random_band_signal(dt, t, W, seed=20130101)
     
        # Filter the input signal u
        v_proper = dt*fftconvolve(u,h,'valid') # convolve u with h
        u_proper = u[len(h)-1:]                # get the proper part of v
        t_proper = t[len(h)-1:]                # get the corresponding time vector
    
        # Encode the filter output v=u*h with an IAF neuron
        spk_train, vol = cim.iaf_encode(dt, t_proper, v_proper, b=bias, d=delta, C=kappa)
        # Identify the filter projection Ph
        h_hat, windows = cim.cim_ideal_iaf(dt, t_Ph, t_proper, u_proper, W, 
                             bias, delta, kappa, spk_train,tau, N_max)
        # Normalized RMSE between h and h_hat
        h_hhat_err   = np.abs(h_long-h_hat)/max(abs(h_long));               # compute the absolute error 
        h_hhat_RMSE  = sqrt(dt*np.trapz(h_hhat_err[idx]**2)/(tau[1]-tau[0]))
        # Compute the RMSE in dB
        MSE_BW[i] = 10*log10( h_hhat_RMSE**2 )
        
    run_time('Fig. 6b time: ',tic_fig6b)       # display the running time for Fig.6b

    # Plot FIg. 6
    fig = p.figure( num='NIPS 2010 Fig. 6',figsize=(12,9), dpi=300, facecolor='w')
    # Plot Fig, 6-a
    ax = fig.add_subplot(2,1,1,xlim=(1,MSE_N.shape[1]),
             title = 'MSE$(\hat{h},\mathcal{P}h)$ vs. the Number of Temporal Windows',
             xlabel = 'Number of Windows $N$',ylabel = 'MSE$(\hat{h},\mathcal{P}h)$, [dB]') 
    lineSpec = ['-o','-s','-d'];linecolor = ['k','b','r'];Marker_Size = [5, 5, 7]
    for d,mse,spec,color,marker_size in zip(D,MSE_N,lineSpec,linecolor,Marker_Size):
        ax.plot(xrange(1,len(mse)+1),mse,spec,color=color,markersize=marker_size,
                label='$D = %d$ Hz' % int(d))
        ax.axvline(x=2*np.pi*100/(np.pi*d), ymin=-100, ymax=40,color=color,
                   linestyle='--',label='$\Omega/(%d \pi)$' % d)
    p.legend()
    # Plot Fig, 6-b
    ax = fig.add_subplot(2,1,2,xlim=(10,150),ylim=(-70,0),xticks = F,
             title  = 'MSE$(\hat{h},h)$ vs. input signal bandwidth',
             xlabel = 'Input signal bandwidth $\Omega/(2\pi)$, [Hz]',
             ylabel = 'MSE$(\hat{h},h)$, [dB]') 
    ax.plot( F, MSE_BW, '-bo', markersize = 5,
             label='$D = 60\;$Hz, $N = %d \quad$' % N_max)
    p.savefig('nips_fig_6.png',dpi=300)
    
    # Finalize the demo
    # -------------------
    run_time('Total Fig. 6 time: ',tic_fig6);           # display the running time for Fig.6
    run_time('Demo time: ', tic_demo);                  # display the demo time