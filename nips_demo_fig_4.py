

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

t0=time.clock()
# Initialize the demo
dt = 1e-6
f  = 25
W  = 2*np.pi*f
Ts = np.pi/W
t  = np.arange(0.,1.5+dt,dt)
t  = t[:2**int(round(math.log(len(t),2)))]
# Create a bandlimited stimulus. The bandwidth W = 2\pi*25 rad/s
# --------------------------------------------------------------
np.random.seed(19871127)                 # fix the state of randn() for reproducible results
fs = 1/dt/len(t)                         #
fu = np.random.randn(int(f/fs)) + \
     np.random.randn(int(f/fs))*1j       # 
u = np.real(np.fft.ifft(fu,len(t)))      #
u = u/max(abs(u))


p.figure(0)  
p.plot(t,u); p.xlabel('time, sec'); p.title('Input Stimulus $$u(t)')
p.show()

# Specify the filter h to be used
# -------------------------------
# Generate a filter according to Adelson and Bergen in [2]. h has a
# temporal support on the interval [T_1, T_2]

T_1 = 0; T_2 = 0.1;                         # specify T_1 and T_2
t_filt = np.arange(T_1,T_2+dt,dt)              # set the length of the impulse response, [s]
a = 200;                                    # set the filter parameter
h = 3*a*np.exp(-a*t_filt)*((a*t_filt)**3/math.factorial(3)\
    -(a*t_filt)**5/math.factorial(5))
# Plot the filter
p.figure(1);
p.plot(t_filt, h);
p.xlabel('Time, [s]');p.ylabel('Amplitude');
p.title('Impulse response $h(t)$ of the filter');
p.show()

# Compute the filter projection Ph
# --------------------------------
# Ph is the projection of h onto the input signal space. It is the best
# approximation of h that can be recovered.

t_Ph = t - (t[1]+t[-1])/2                 # get the time vector for Ph
g = W/np.pi*np.sinc(W*t_Ph/np.pi)         # calculate the sinc kernel g  
Ph = dt*fftconvolve(h,g)[:len(t_Ph)]      # find the projection Ph by convolving h with g

# Plot the filter projection
p.figure(2)
p.plot(t_Ph, Ph)
p.xlabel('Time, [s]')
p.ylabel('Amplitude')
p.xlim([-0.05, 0.15])
p.title('Filter projection $\mathcal{P}h(t)$')
p.show()


# Filter the input signal u
# -------------------------
# Since all signals are finite, the filter output v=u*h is not calculated
# properly on the boundaries. v_proper is that part of v, for which the
# convolution of u and h is computed correctly.

v = dt*fftconvolve(h,u)[:len(u)]                        # convolve u with h

v_proper = v[len(h):]                # get the proper part of v
t_proper = t[len(h):]                # get the correspoding time vector
u_proper = u[len(h):]                # get the corresponding stimulus

# Plot the filter output
p.figure(3)
p.plot(t,v);p.xlabel('Time, [s]');
p.ylabel('Amplitude');
p.title('Filter output $v(t)$');
p.show()
# Encode the filter output v=u*h with an IAF neuron
# -------------------------------------------------
# Specify parameters of the Ideal IAF neuron
delta = 0.007;                              # set the threshold
bias  = 0.28;                               # set the bias
kappa = 1;                                  # set capacitance

spk_train = iaf.iaf_encode(v_proper, dt, b=bias, d=delta,C=kappa)
spk_train = np.cumsum(spk_train)



T_filt_rec = 0.12;                          # specify the hypothesized length of the impulse response 
tau = [-(T_filt_rec - t_filt[-1])/2, 0]     # get tau_1 and tau_2. Here W is centered around the actual
tau[1] = t_filt[-1]-tau[0];                 # temporal support T_2-T_1
N_max = 10;                                 # set the maximum number of windows to be used (could be smaller depending on the simulation)

[h_hat, windows] = cim.cim_ideal_iaf(dt, t_Ph, t_proper, u_proper, W, bias, delta, kappa, 
                                     spk_train+t_proper[0],tau, N_max)
# Compute MMSE between h_hat and h, and MMSE between h_hat and Ph
h_long = np.zeros_like(t_Ph)
h_long[ [i for i in xrange(len(t_Ph)) if t_filt[0] <= t_Ph[i] and t_Ph[i] <= t_filt[-1] ] ] = h

t_frame_idx = [i for i in xrange(len(t_Ph)) if tau[0] <= t_Ph[i] and t_Ph[i] <= tau[1] ]    # find indices of t for the given frame

# Normalized RMSE between h and h_hat
h_hhat_err   = np.abs(h_long-h_hat)/max(abs(h_long));               # compute the absolute error 
h_hhat_RMSE  = math.sqrt(dt*np.trapz(h_hhat_err[t_frame_idx]**2)/     # compute the RMSE
                       (len(t_frame_idx)*dt));
                   
# Normalized RMSE between Ph and h_hat
Ph_hhat_err  = np.abs(Ph-h_hat)/max(abs(Ph));                       # compute the absolute error
Ph_hhat_RMSE = math.sqrt(dt*np.trapz(Ph_hhat_err[t_frame_idx]**2)/    # compute the RMSE
                       (len(t_frame_idx)*dt)); 

p.figure(4)
palette = ['r','g','b','y','m','c']
for i in xrange(len(windows)):
    p.axvspan(windows[i][0]-t_proper[0],windows[i][1]-t_proper[0],\
              facecolor=palette[i%6],edgecolor='none')
p.stem(spk_train, np.ones_like(spk_train),linefmt='k-', markerfmt='k^')
p.ylim([0, 1.1])
p.yticks([])
p.xlabel('Time, [s]');
p.ylabel('Spike')
p.show()


p.figure(5)
p.plot(t_Ph,h_hat,'-b',t_Ph,Ph,'g')
p.xlim([-0.05, 0.15])
p.title('Identified filter $\hat{h}(t)$')
p.xlabel('Time, [s]')
p.show()

#
fig = p.figure(num='NIPS 2010 Fig. 4',figsize=(20,15),dpi=300, facecolor='w')
fig.suptitle('NIPS 2010 Figure 4\n' +
             'A.A. Lazar and Y.B. Slutskiy',
             fontsize=24)

# plot the input stimulus u
ax = fig.add_subplot(3,2,1,
                     xlim=(0,1),ylim=(-1,1), ylabel='Amplitude', xlabel='Time, [s]',
                     title='$(a)\qquad$Input signal u(t)');
ax.plot(t_proper - t_proper[0], u_proper,
        label='$\Omega = 2\pi\cdot '+str(f)+ '$rad/s$\qquad$')
p.legend(loc='lower right')
#plot the periodogram power spectrum estmate of u
ax = fig.add_subplot(3,2,2,
                     xlim=(-150,150),ylim=(-140,0),#ylabel='Power, [dB]',
                     title='$(d)\qquad$Periodogram Power Spectrum Estimate of u(t)');
ax.psd(u_proper,pad_to=len(t),NFFT=len(t),Fs=1/dt,sides='twosided',label='supp$(\mathcal{F}u)=[-\Omega,\Omega]\qquad$')
p.legend(loc='lower right')
# plot h, Ph and h_hat (the filter identified by the algorithm)
ax = fig.add_subplot(3,2,3,
                     xlim=(0,1),ylim=(0,1.2), xlabel='Time, [s]',
                     yticks=[],yticklabels=[],
                     title='$(b)\qquad$Output of the [Filter]-[Ideal IAF] neural circuit');
palette = ['r','g','b','y','m','c']
ax.stem(spk_train, np.ones_like(spk_train),linefmt='k-', markerfmt='k^',label='$D = 40$Hz')
for i,win in enumerate(windows):
    ax.axvspan(win[0]-t_proper[0],win[1]-t_proper[0],\
               facecolor=palette[i%6],edgecolor='none',
               label='_nolegend_')
p.legend(loc='lower right')
# plot the periodogram power spectrum estmate of h
ax = fig.add_subplot(3,2,4,
                     xlim=(-150,150),ylim=(-140,0),#ylabel='Power, [dB]',
                     title='$(e)\qquad$Periodogram Power Spectrum Estimate of h(t)');
ax.psd(h,pad_to=len(t),NFFT=len(t),Fs=1/dt,sides='twosided',
       label='supp$(\mathcal{F}h)\supset[-\Omega,\Omega]\qquad$')
p.legend(loc='lower right')
# plot the periodogram power spectrum estmate of v=u*h
ax = fig.add_subplot(3,2,5,
                     xlim=(-0.05,0.15), ylim=(-50,100), ylabel='Amplitude', xlabel='Time, [s]',
                     title='$(c)\qquad$Original filter vs. the identified filter');
ax.plot(t_Ph, h_long,'--k',label='$h,\,$RMSE$(\hat{h},h)$ = %3.2e $\qquad$' % h_hhat_RMSE )
ax.plot(t_Ph, Ph,'-b',label='$\mathcal{P}h,\,$RMSE$(\hat{h},\mathcal{P}h)$ = %3.2e $\qquad$' % Ph_hhat_RMSE)
ax.plot(t_Ph,h_hat,'-r',label='$\hat{h}$')
p.legend(loc='upper right')

# plot the periodogram power spectrum estmate of v=u*h
ax = fig.add_subplot(3,2,6,
                     xlim=(-150,150),ylim=(-140,0),#ylabel='Power, [dB]',
                     title='$(f)\qquad$Periodogram Power Spectrum Estimate of v(t)');
ax.psd(v_proper,pad_to=len(t),NFFT=len(t),Fs=1/dt,sides='twosided',label='supp$(\mathcal{F}v)=[-\Omega,\Omega]\qquad$')
p.legend(loc='lower right')
fig.text(.1,0.05,'[1] $Identifying\ Dendritic\ Processing$, ' +
               'Advances in Neural Information Processing Systems 23, pp. 1261-1269, 2010')
p.savefig('nips_fig_4.png',dpi=300)
print "Run time: %s" % repr(time.clock()-t0)
