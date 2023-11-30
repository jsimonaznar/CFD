import matplotlib.pyplot as plt
import numpy as np

## INPUT DATA
# Numeric
N        = [20,100,20,100] # Number of Fourier modes
Ck       = [0.05,0.05, 0.05, 0.05]  # Kolmogorov
C        = 0.02  # Constant for explicit scheme integration, 0<C<1
time_max = 100000000000
delta      = 1.0E-06 # Accuracy
m        = 2 
LES      = 1 # If LES=1 we compute LES, if not DNS

# Physical
L  = 1.0 # Characteristic lenght
U  = 1.0 # Characteristic viscosity  
nu = [1.0/10.0, 1.0/10.0, 1.0/70.0, 1.0/70.0]  # Kinematic viscosity, Re in denominator
i  = 1j

# Solver
for Z in range(len(N)):
    delta_T  = (C * (1.0/nu[Z])) / N[Z]**2 # Time step

    uk      = np.zeros(N[Z], dtype=complex)
    uk0 = np.zeros(N[Z], dtype=complex)
    Ek      = np.zeros(N[Z])      

    # Only k>0 values
    uk[0]      = 0.0
    uk0[0] = 0.0
    uk[1]      = 1.0
    uk0[1] = 1.0
    for k in range(2, N[Z]):
        uk0[k] = 1.0 / float(k+1)
      
    # Begin time loop
    for time in range(time_max):
        t0 = np.abs(np.sum(uk0))
        for k in range(2, N[Z]):

            # Convective term
            conv = 0 
            for p in range(-N[Z]+1,N[Z]):
                q = k - p
                if (q < -N[Z] + 1) or (q >= N[Z]):
                    uq = 0.0
                    up = 0.0
                else:
                    uq = np.conjugate(uk0[np.abs(q)]) if q < 0 else uk0[q]
                    up = np.conjugate(uk0[np.abs(p)]) if p < 0 else uk0[p]
                conv = conv + q * i * uq * up

            # n+1 velocity
            if LES == 0:
                uk[k] = uk0[k] - delta_T * (nu[Z] * k**2 * uk0[k] + conv)
            elif LES == 1 :
                nu_inf = 0.31 * ((5 - m) / (m + 1)) * np.sqrt(3 - m) * Ck[Z] ** (-3.0 / 2.0)
                EkN    = (uk0[-1] * np.conjugate(uk0[-1])).real
                nu_a   = 1.0 + 34.5 * np.exp(-3.03 * (N[Z] / k) )
                nu_t   = nu_inf * ((EkN / N[Z]) ** (0.5) * nu_a)
                nu_eff = nu[Z] + nu_t
                uk[k] = uk0[k] - delta_T * (nu_eff * k**2 * uk0[k] + conv)

        t = np.abs(np.sum(uk))
        if np.abs(t-t0) < delta:
            break
        else:
            uk0 = uk
    
    # Energy spectrum
    for k in range(N[Z]):
        Ek[k] = (uk[k] * np.conjugate(uk[k])).real
    if Z == 0 and LES == 1:
        k_cv1 = np.linspace(1, N[0], N[0])
        Ek1   = np.zeros(N[0])
        Ek1   = Ek
    elif Z == 1 and LES == 1:
        k_cv2 = np.linspace(1, N[1], N[1])
        Ek2 = np.zeros(N[1])
        Ek2 = Ek
    elif Z == 2 and LES == 1:
        k_cv3 = np.linspace(1, N[0], N[0])
        Ek3   = np.zeros(N[0])
        Ek3   = Ek
    else:
        k_cv4 = np.linspace(1, N[1], N[1])
        Ek4 = np.zeros(N[1])
        Ek4 = Ek

## PLOT
# Reference lines
X = np.linspace(1, N[1], N[1])
Y = np.linspace(1, N[1], N[1])
for i in range(1,N[1]):
    Y[i] = (X[i])**-2

# Figure specifications
fontsize=15

# Start plot
plt.figure(1)
plt.figure(figsize = (10,8))
plt.tick_params(axis='both', which='both',length=3, width=1.0,
labelsize=15, right=True, top=True, direction='in') 

# Labels
plt.ylabel(r"$E_{k}$", fontsize=fontsize)
plt.xlabel(r"$k$", fontsize=fontsize)

# Limits
plt.xlim(k_cv1[1], N[1])

# Grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Plot
plt.loglog(k_cv1[1:], Ek1[1:],'-*', color='red', label=r"LES - $Re = 10 ~ and ~ N = 20$")
plt.loglog(k_cv2[1:], Ek2[1:], '-o', color='blue', label=r"LES - $Re = 10 ~ and ~ N = 100$")
plt.loglog(k_cv3[1:], Ek3[1:],'-+', color='yellow', label=r"LES - $Re = 70 ~ and ~ N = 20$")
plt.loglog(k_cv4[1:], Ek4[1:], '-x', color='green', label=r"LES - $Re = 70 ~ and ~ N = 100$")
plt.plot(X + 1, Y, linestyle="dashed", color="black", label=r"$Slope=-2$")

# Legend 
plt.legend(fontsize=12, loc='upper right')

plt.show()
plt.close(1)