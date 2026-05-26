import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. PREP DATA ---
df3 = pd.read_csv('data_3.csv')
df3_after = pd.read_csv('data_3_after_t0.csv')
df = pd.concat([df3, df3_after]).sort_values('time_s').reset_index(drop=True)

r = df['range_m'].values
theta = np.deg2rad(df['bearing_deg'].values)
x_meas = r * np.cos(theta)
y_meas = r * np.sin(theta)

dt = 1.0
n_steps = len(df)

F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1,  0],
    [0, 0, 0,  1]
])

def get_Q(q_val):
    return q_val * np.array([
        [dt**3/3, 0,       dt**2/2, 0],
        [0,       dt**3/3, 0,       dt**2/2],
        [dt**2/2, 0,       dt,      0],
        [0,       dt**2/2, 0,       dt]
    ])

# Radar R
sigma_r = 50.0  
sigma_theta = np.deg2rad(2.0)
R_radar = np.diag([sigma_r**2, sigma_theta**2])

# Base functions
def ekf_predict(x, P, F, Q):
    return F @ x, F @ P @ F.T + Q

def ekf_update_radar(x_pred, P_pred, z, R):
    px, py, vx, vy = x_pred.flatten()
    r_est = max(np.sqrt(px**2 + py**2), 1e-4)
    theta_est = np.arctan2(py, px)
    z_pred = np.array([[r_est], [theta_est]])
    
    y = z - z_pred
    y[1, 0] = (y[1, 0] + np.pi) % (2 * np.pi) - np.pi
    
    H = np.array([
        [px/r_est, py/r_est, 0, 0],
        [-py/(r_est**2), px/(r_est**2), 0, 0]
    ])
    
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    x_upd = x_pred + K @ y
    P_upd = (np.eye(4) - K @ H) @ P_pred
    return x_upd, P_upd, y, S

def rts_smoother(x_upd_list, P_upd_list, x_pred_list, P_pred_list, F):
    n = len(x_upd_list)
    x_smooth = [np.zeros_like(x_upd_list[0]) for _ in range(n)]
    P_smooth = [np.zeros_like(P_upd_list[0]) for _ in range(n)]
    
    x_smooth[-1] = x_upd_list[-1]
    P_smooth[-1] = P_upd_list[-1]
    
    for k in range(n - 2, -1, -1):
        P_pred = P_pred_list[k+1]
        try:
            inv_P_pred = np.linalg.inv(P_pred)
        except:
            inv_P_pred = np.linalg.pinv(P_pred)
            
        C = P_upd_list[k] @ F.T @ inv_P_pred
        
        x_smooth[k] = x_upd_list[k] + C @ (x_smooth[k+1] - x_pred_list[k+1])
        P_smooth[k] = P_upd_list[k] + C @ (P_smooth[k+1] - P_pred) @ C.T
        
    return x_smooth

x0 = np.array([[x_meas[0]], [y_meas[0]], [(x_meas[1]-x_meas[0])/dt], [(y_meas[1]-y_meas[0])/dt]])
P0 = np.eye(4) * 1000.0

# --- BASELINE: EKF + RTS ---
def run_ekf(x0, P0, Q_val):
    Q = get_Q(Q_val)
    x_upd, P_upd, x_pred, P_pred = [], [], [], []
    xc, Pc = np.copy(x0), np.copy(P0)
    for i in range(n_steps):
        z = np.array([[r[i]], [theta[i]]])
        xp, Pp = ekf_predict(xc, Pc, F, Q)
        xu, Pu, _, _ = ekf_update_radar(xp, Pp, z, R_radar)
        x_pred.append(xp); P_pred.append(Pp)
        x_upd.append(xu); P_upd.append(Pu)
        xc, Pc = xu, Pu
    return x_upd, P_upd, x_pred, P_pred

b_xu, b_pu, b_xp, b_pp = run_ekf(x0, P0, 5.0)
baseline_smooth = rts_smoother(b_xu, b_pu, b_xp, b_pp, F)

# --- EXTRACT ---
def get_xy(x_list):
    return [x[0,0] for x in x_list], [x[1,0] for x in x_list]

# EKF only (updated states)
ekf_x, ekf_y = get_xy(b_xu)

# EKF + RTS
rts_x, rts_y = get_xy(baseline_smooth)

# ==========================================================
# PLOT 1 : EKF ONLY
# ==========================================================
plt.figure(figsize=(5, 3.5))

plt.plot(
    x_meas, y_meas,
    'k.', alpha=0.3,
    label='Raw Radar Data'
)

plt.plot(
    ekf_x, ekf_y,
    'b-', linewidth=2,
    label='EKF'
)

plt.title('EKF Trajectory', fontsize=10)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.savefig('ekf_only_data_3.png', dpi=100, bbox_inches='tight')

# ==========================================================
# PLOT 2 : EKF + RTS
# ==========================================================
plt.figure(figsize=(5, 3.5))

plt.plot(
    x_meas, y_meas,
    'k.', alpha=0.3,
    label='Raw Radar Data'
)

plt.plot(
    rts_x, rts_y,
    'r-', linewidth=2.5,
    label='EKF + RTS'
)

plt.title('EKF + RTS Smoothed Trajectory', fontsize=10)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.savefig('ekf_rts_data_3.png', dpi=100, bbox_inches='tight')

plt.show()  