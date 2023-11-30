import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

## INPUT DATA

# Numerical
N        = [100,100,100,100] # Vector length
Ck       = [0.4523,0.05]  # Ck parameter
C        = 0.02           # Cosntant for explicit scheme integration, 0<C<1
time_max = 100000000000   # Max loop time
eps      = 1.0E-06        # Convergence parameter
m        = 2              # Slope
LES      = 0              # If LES=1 we compute LES

# Physical
L  = 1.0                  # Characteristic lenght
U  = 1.0                  # Characteristic viscosity  
Re = 40.0                 # Reynolds number
nu = (U * L) / Re         # Kinematic viscosity
i  = 1j                   # Complex number i

# Solver
for Z in range(len(N)):
    if Z==2:
        LES = 1
    delta_T = (C * Re) / N[Z]**2  # Time step

    uk = np.zeros(N[Z], dtype=complex)  # u_k vectors as complex
    uk0 = np.zeros(N[Z], dtype=complex)  # u_k at the first time step
    Ek = np.zeros(N[Z])  # Energy spectrum

    # Initialize velocities
    # Only k>0 values
    uk[0] = 0.0
    uk0[0] = 0.0
    uk[1] = 1.0
    uk0[1] = 1.0
    for k in range(2, N[Z]):
        uk0[k] = 1.0 / float(k+1)

    # Begin time step (t+delta_t)
    for time in range(time_max):
        t0 = np.abs(np.sum(uk0))
        for k in range(2, N[Z]):

            # Convective term first
            conv = 0 
            for p in range(-N[Z]+1,N[Z]):
                q = k - p
                if (q < -N[Z] + 1) or (q >= N[Z]): # Out of domain
                    uq = 0.0
                    up = 0.0
                else:
                    uq = np.conjugate(uk0[np.abs(q)]) if q < 0 else uk0[q]
                    up = np.conjugate(uk0[np.abs(p)]) if p < 0 else uk0[p]
                conv = conv + q * i * uq * up
            if LES == 0:
                uk[k] = uk0[k] - delta_T * (nu * k**2 * uk0[k] + conv)
            elif LES == 1 :
                nu_inf = 0.31 * ((5 - m) / (m + 1)) * np.sqrt(3 - m) * Ck[Z-2] ** (-3.0 / 2.0)
                EkN = (uk0[-1] * np.conjugate(uk0[-1])).real
                nu_a = 1.0 + 34.5 * np.exp(-3.03 * (N[Z] / k) )
                nu_t = nu_inf * ((EkN / N[Z]) ** (0.5) * nu_a)
                nu_eff = nu + nu_t
                uk[k] = uk0[k] - delta_T * (nu_eff * k**2 * uk0[k] + conv)

        t = np.abs(np.sum(uk))
        if np.abs(t-t0) < eps:
            print(f"Time loop converged for N={N[Z]}")
            break
        else:
            uk0 = uk

    # Energy spectrum
    for k in range(N[Z]):
        Ek[k] = (uk[k] * np.conjugate(uk[k])).real
    if Z == 2 and LES == 1:
        k_cv1LES = np.linspace(1, N[0], N[0])
        Ek1LES = np.zeros(N[0])
        Ek1LES = Ek
    elif Z == 3 and LES == 1:
        k_cv2LES = np.linspace(1, N[1], N[1])
        Ek2LES = np.zeros(N[1])
        Ek2LES = Ek
    elif Z == 0 and LES == 0 :
        k_cv1 = np.linspace(1, N[0], N[0])
        Ek1 = np.zeros(N[0])
        Ek1 = Ek
    else:
        k_cv2 = np.linspace(1, N[1], N[1])
        Ek2 = np.zeros(N[1])
        Ek2 = Ek

# PLOT
# Create the references line
X = np.linspace(1, N[1], N[1])
Y = np.linspace(1, N[1], N[1])
for i in range(1,N[1]):
    Y[i] = (X[i])**-2

# Figure specifications
fontsize=15

# Start first plot
plt.figure(1)
plt.figure(figsize = (10,8))
plt.tick_params(axis='both', which='both',length=3, width=1.0,
labelsize=15, right=True, top=True, direction='in') # For ticks in borders

# Figure labels
plt.ylabel(r"$E_{k}$", fontsize=fontsize)
plt.xlabel(r"$k$", fontsize=fontsize)

# Limits
plt.xlim(k_cv1[1], N[1])
plt.ylim(10E-07,1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Plot
#plt.loglog(k_cv1[1:], Ek1[1:], color='blue', label=r"$N=20$ DNS")
plt.loglog(k_cv2[1:], Ek2[1:],  color='blue', label=r"$N=100$")
plt.loglog(k_cv1LES[1:], Ek1LES[1:], color='red', label=r"$N=100$ LES $C_{K}=0.4523$")
plt.loglog(k_cv1LES[1:], Ek2LES[1:],  color='green', label=r"$N=100$ LES $C_{K}=0.05$")
plt.plot(X + 1, Y, linestyle="dashed", color="black", label=r"$Slope=-2$")

plt.legend(fontsize = 12, loc='upper right')

plt.show()

plt.close(1)