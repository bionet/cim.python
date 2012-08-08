
# Get data for Fig. 6a
#
# To demonstrate how the MSE changes as a function of the neuron spike
# density D, we encode the input signal with an IAF neuron having a
# different bias b = D*delta/kappa.
#Fig6a_tic = tic;                            # initialize the timer

import bionet.ted.iaf as iaf
import cim
import scipy
from scipy.signal import fftconvolve
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as p
import math
import time


# Initialize the demo
dt = 1e-6
f  = 100
W  = 2*np.pi*f
Ts = np.pi/W
t  = np.arange(0.,8.5,dt)
t  = t[:2**int(round(math.log(len(t),2)))]
# Create a bandlimited stimulus. The bandwidth W = 2\pi*25 rad/s
# --------------------------------------------------------------
np.random.seed(19871127);                # fix the state of randn() for reproducible results
#u = np.zeros_like(t)                     # initialize the stimulus u(t)
#N_samp = int((t[-1]-t[0])/Ts);           # find the number of stimulus samples
t0=time.clock()
#for k in xrange(1,N_samp+1):
#    u += np.random.randn()*np.sinc(W*(t-k*Ts)/np.pi)  # the first sample is zero
#u = u/max(abs(u))
np.random.seed(19871127)                 # fix the state of randn() for reproducible results
fs = 1/dt/len(t)                         #
fu = np.random.randn(int(f/fs)) + \
     np.random.randn(int(f/fs))*1j       # 
u = np.real(np.fft.ifft(fu,len(t)))      #
u = u/max(abs(u))
print 'u(t) synthesizing time:%s' % repr(time.clock()-t0)

# Specify the filter h to be used
# -------------------------------
# Generate a filter according to Adelson and Bergen in [2]. h has a
# temporal support on the interval [T_1, T_2]

T_1 = 0; T_2 = 0.1;                         # specify T_1 and T_2
t_filt = np.arange(T_1,T_2,dt)              # set the length of the impulse response, [s]
a = 200;                                    # set the filter parameter
h = 3*a*np.exp(-a*t_filt)*((a*t_filt)**3/math.factorial(3)\
    -(a*t_filt)**5/math.factorial(5))

# Compute the filter projection Ph
# --------------------------------
# Ph is the projection of h onto the input signal space. It is the best
# approximation of h that can be recovered.

t_Ph = t - (t[1]+t[-1])/2                 # get the time vector for Ph
g = W/np.pi*np.sinc(W*t_Ph/np.pi)         # calculate the sinc kernel g  
Ph = dt*fftconvolve(h,g)[:len(t_Ph)]      # find the projection Ph by convolving h with g

# Filter the input signal u
# -------------------------
# Since all signals are finite, the filter output v=u*h is not calculated
# properly on the boundaries. v_proper is that part of v, for which the
# convolution of u and h is computed correctly.

v = dt*fftconvolve(h,u)[:len(u)]                        # convolve u with h

v_proper = v[len(h):]                # get the proper part of v
t_proper = t[len(h):]                # get the corresponding time vector
u_proper = u[len(h):]                # get the corresponding stimulus

# Encode the filter output v=u*h with an IAF neuron
# -------------------------------------------------
# Specify parameters of the Ideal IAF neuron
delta = 0.007;                              # set the threshold
kappa = 1;                                  # set capacitance


T_filt_rec = 0.11;                          # specify the hypothesized length of the impulse response 
tau = [-(T_filt_rec - t_filt[-1])/2, 0]     # get tau_1 and tau_2. Here W is centered around the actual
tau[1] = t_filt[-1]-tau[0];                 # temporal support T_2-T_1

D = [20, 40, 60]                            # set the average spiking density
MSE_palette = 'kbr';                        # set the color palette for plotting
MSE_N = []                                  # initialize the MSE cell
N_max = 30;                                 # set the maximum number of windows to be used

t_frame_idx = [i for i in xrange(len(t_Ph)) if tau[0] <= t_Ph[i] and t_Ph[i] <= tau[1] ]    # find indices of t for the given frame

for d in D:
    
    bias  = d*delta/kappa;         # set the bias
    # Encode the filter output v=u*h with an IAF neuron
    t0=time.clock()
    spk_train = iaf.iaf_encode(v_proper, dt, b=bias, d=delta,C=kappa)
    spk_train = np.cumsum(spk_train)+t_proper[0]
    print "Number of spikes: %d" % len(spk_train)
    print 'Encoding time:%s' % repr(time.clock()-t0)
    # Identify the filter projection Ph, calculate MSE as a function of N
    t0=time.clock()
    [h_hat, windows, h_hat_N] = cim.cim_ideal_iaf(dt, t_Ph, t_proper, u_proper, W, bias, delta, kappa,\
                                     spk_train,tau, N_max, Calc_MSE_N=True)
    print 'Number of window: %d' % len(windows)
    print 'Decoding time:%s' % repr(time.clock()-t0)

    N = len(windows)
    # Compute the normalized MSE
    mse = np.zeros(N) 
    Ph_hhat_err  = abs(Ph-h_hat)/max(abs(Ph));
    Ph_hhat_RMSE = math.sqrt(dt*np.trapz(Ph_hhat_err[t_frame_idx]**2)/ 
                                 (len(t_frame_idx)*dt))
    #print Ph_hhat_RMSE
    
    for i in xrange(N):
        # Normalized RMSE error for Ph - h_hat
        Ph_hhat_err  = abs(Ph-h_hat_N[i])/max(abs(Ph));
        Ph_hhat_RMSE = math.sqrt(dt*np.trapz(Ph_hhat_err[t_frame_idx]**2)/ 
                                 (len(t_frame_idx)*dt));
        mse[i] = 10*math.log10( Ph_hhat_RMSE**2 );
        #print 'MMSE-%d: %s' % (N,mse[i])

    MSE_N.append( mse )


# Get data for Fig. 6b
# --------------------
# In the following, for fixed number of windows N, we compute the MSE
# between the original filter h and the identified filter h_hat as a
# function of the input signal bandwidth
        
F = np.arange(10,160,10)                    # specify the bandwidth vector
t = np.arange(0,2.0,dt)
t = t[:2**int(round(math.log(len(t),2)))]   # specify the time vector
N_max = 10;                                 # set the number of windows

# Compute the filter projection Ph
t_Ph = t - (t[1]+t[-1])/2                   # get the time vector for Ph
g = W/np.pi*np.sinc(W*t_Ph/np.pi)           # calculate the sinc kernel g  
Ph = dt*fftconvolve(h,g)[:len(t_Ph)]        # find the projection Ph by convolving h with g

# h_long is a zero-padded version of h (of the same size as t_Ph)
h_long = np.zeros_like(t_Ph)
h_long[ [i for i in xrange(len(t_Ph)) if t_filt[0] <= t_Ph[i] and t_Ph[i] <= t_filt[-1] ] ] = h
t_frame_idx = [i for i in xrange(len(t_Ph)) if tau[0] <= t_Ph[i] and t_Ph[i] <= tau[1] ]    # find indices of t for the given frame


# specify Ideal IAF neuron parameters
delta = 0.007;                              # set the threshold
bias  = 0.42;                               # set the bias
kappa = 1;                                  # set the capacitance


MSE_BW = np.zeros_like(F)
    
for idx,f in enumerate(F):
    W   = 2*np.pi*f                         # calculate the bandwidth in [rad]
    Ts  = np.pi/W;                          # calculate the sampling period in [s]
    
    
    # Create a bandlimited signal
    np.random.seed(19871127)                 # fix the state of randn() for reproducible results
    fs = 1/dt/len(t)                         #
    fu = np.random.randn(int(f/fs)) + \
         np.random.randn(int(f/fs))*1j       # 
    u = np.real(np.fft.ifft(fu,len(t)))      #
    u = u/max(abs(u))
#
    v = dt*fftconvolve(h,u)[:len(u)]         # convolve u with h

    v_proper = v[len(h):]                    # get the proper part of v
    t_proper = t[len(h):]                    # get the correspoding time vector
    u_proper = u[len(h):]                    # get the corresponding stimulus

    # Encode the filter output v=u*h with an IAF neuron
    spk_train = iaf.iaf_encode(v_proper, dt, b=bias, d=delta,C=kappa)
    spk_train = np.cumsum(spk_train)

    h_hat, windows = cim.cim_ideal_iaf(dt, t_Ph, t_proper, u_proper, W, bias, delta, kappa,\
                                       spk_train+t_proper[0],tau, N_max)
    
    # Normalized RMSE between h and h_hat
    h_hhat_err   = np.abs(h_long-h_hat)/max(abs(h_long));               # compute the absolute error 
    h_hhat_RMSE  = math.sqrt(dt*np.trapz(h_hhat_err[t_frame_idx]**2)/   # compute the RMSE
                             (len(t_frame_idx)*dt));
    
    MSE_BW[idx] = 10*np.log10( h_hhat_RMSE**2 )

# Plot Fig. 6 of [1]
fig = p.figure( num='NIPS 2010 Fig. 6',figsize=(12,9), dpi=300, facecolor='w')
# Plot Fig, 6-a
ax = fig.add_subplot(2,1,1,
                     xlim=(0,30),
                     title = 'MSE$(\hat{h},\mathcal{P}h)$ vs. the Number of Temporal Windows',
                     xlabel = 'Number of Windows $N$',ylabel = 'MSE$(\hat{h},\mathcal{P}h)$, [dB]',
                     ) 

lineSpec = ['-o','-s','-d']
linecolor = ['k','b','r']
Marker_Size = [5, 5, 7]
for d,mse,spec,color,marker_size in zip(D,MSE_N,lineSpec,linecolor,Marker_Size):
    ax.plot(xrange(len(mse)),mse,spec,
            color=color,
            markersize=marker_size, 
            label='$D = %d\_$Hz' % int(d))
    ax.axvline(x=np.pi*100/(np.pi*d), ymin=-100, ymax=40,
               color=color,
               linestyle='--',
               label='$\Omega/(%d \pi)$' % d)
p.legend()
# Plot Fig, 6-b
ax = fig.add_subplot(2,1,2,
                     xlim=(10,150),ylim=(-70,0),
                     title = 'MSE$(\hat{h},h)$ vs. input signal bandwidth',
                     xlabel = 'Input signal bandwidth $\Omega/(2\pi)$, [Hz]',
                     ylabel = 'MSE$(\hat{h},h)$, [dB]',
                     xticks = F
                     ) 
ax.plot( F, MSE_BW, '-bo', markersize = 5,
      label='$D = 60\;$Hz, $N = %d \quad$' % N_max)
p.legend()
p.savefig('nips_fig_6.png',dpi=300)



