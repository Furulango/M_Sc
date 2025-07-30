import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def motor_induccion(t, x, params, vqs, vds):
    """
    Modelo básico motor inducción en coordenadas DQ
    x = [iqs, ids, iqr, idr, wr] - estados
    """
    iqs, ids, iqr, idr, wr = x
    rs, rr, Lls, Llr, Lm, J, B = params
    
    # Inductancias totales
    Ls, Lr = Lls + Lm, Llr + Lm
    we = 2*np.pi*60  # 60 Hz
    ws = we - wr     # Deslizamiento
    
    # Flujos magnéticos
    lqs = Ls*iqs + Lm*iqr
    lds = Ls*ids + Lm*idr
    lqr = Lr*iqr + Lm*iqs
    ldr = Lr*idr + Lm*ids
    
    # Ecuaciones de voltaje (matriz L * di/dt = v)
    L = np.array([[Ls, 0, Lm, 0], [0, Ls, 0, Lm], 
                  [Lm, 0, Lr, 0], [0, Lm, 0, Lr]])
    v = np.array([vqs - rs*iqs - we*lds, vds - rs*ids + we*lqs,
                  -rr*iqr - ws*ldr, -rr*idr + ws*lqr])
    
    di_dt = np.linalg.solve(L, v)
    
    # Par electromagnético y ecuación mecánica
    Te = (3*4/4) * Lm * (iqs*idr - ids*iqr)  # P=4 polos
    dwr_dt = (Te - B*wr) / J
    
    return np.array([*di_dt, dwr_dt])

# Parámetros del motor
params = [2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001]

# Voltajes DQ (arranque con 220V)
vqs, vds = 220*np.sqrt(2)/np.sqrt(3), 0  # Voltaje pico de fase

# Simular 2 segundos, condiciones iniciales cero
sol = solve_ivp(lambda t, x: motor_induccion(t, x, params, vqs, vds),
                [0, 2], [0,0,0,0,0], dense_output=True)

# Resultados
t = np.linspace(0, 2, 1000)
iqs, ids, iqr, idr, wr = sol.sol(t)

# Par electromagnético y RPM
Te = (3*4/4) * 0.203 * (iqs*idr - ids*iqr)
rpm = wr * 60/(2*np.pi) * 2/4

# Gráficar
fig, ax = plt.subplots(2, 2, figsize=(10, 6))
ax[0,0].plot(t, iqs, 'b', label='iqs'); ax[0,0].plot(t, ids, 'r', label='ids')
ax[0,0].set_title('Corrientes Estator'); ax[0,0].legend()
ax[0,1].plot(t, np.sqrt(iqs**2 + ids**2), 'k')
ax[0,1].set_title('|Is|')
ax[1,0].plot(t, Te, 'orange'); ax[1,0].set_title('Par (N⋅m)')
ax[1,1].plot(t, rpm, 'purple'); ax[1,1].set_title('Velocidad (RPM)')
plt.tight_layout(); plt.show()

print(f"Valores finales: Is={np.sqrt(iqs[-1]**2 + ids[-1]**2):.2f}A, Te={Te[-1]:.2f}Nm, RPM={rpm[-1]:.0f}")