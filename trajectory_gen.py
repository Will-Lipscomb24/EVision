# TRajectory Gen 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# --- Time ---
dt  = 0.01
t   = np.arange(0, 20 + dt, dt)
N   = len(t)

# --- Testbed / Spiral Params ---
testbed   = 6.0
margin    = 0.2
r_max     = testbed / 2 - margin   # 2.8m
r_t       = np.linspace(0.5, r_max, N)
n_revs    = 1
omega     = 2 * np.pi * n_revs / 20
target    = np.array([0.0, 0.0, 0.0])

# --- Position ---
px = r_t * np.cos(omega * t)
py = r_t * np.sin(omega * t)
pz = np.zeros(N)

# Clamp to testbed
px = np.clip(px, -testbed/2, testbed/2)
py = np.clip(py, -testbed/2, testbed/2)
pos = np.stack([px, py, pz], axis=0)  # 3xN

# --- Linear Velocity (finite diff) ---
vel = np.diff(pos, axis=1) / dt
vel = np.hstack([vel, vel[:, -1:]])

# --- Quaternion (body Z points toward target) ---
def rotm_to_quat(R):
    """Rotation matrix to quaternion [w, x, y, z]"""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

quats = np.zeros((4, N))
for i in range(N):
    p = pos[:, i]
    z_body = target - p
    if np.linalg.norm(z_body) < 1e-6:
        z_body = np.array([1.0, 0.0, 0.0])
    z_body /= np.linalg.norm(z_body)

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_body, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])

    x_body = np.cross(up, z_body)
    x_body /= np.linalg.norm(x_body)
    y_body = np.cross(z_body, x_body)

    R = np.column_stack([x_body, y_body, z_body])
    quats[:, i] = rotm_to_quat(R)

# --- Angular Velocity (finite diff) ---
ang_vel = np.zeros((3, N))
for i in range(N - 1):
    q1 = quats[:, i]
    q2 = quats[:, i + 1]
    dq = (q2 - q1) / dt
    q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    oq = quat_mult(q1_conj, dq)
    ang_vel[:, i] = 2 * oq[1:]
ang_vel[:, -1] = ang_vel[:, -2]


# --- Discretize at 1Hz ---
step = int(1.0 / dt)  # every 100 samples = 1 second
idx  = np.arange(0, N, step)

t_d       = t[idx]
pos_d     = pos[:, idx]
vel_d     = vel[:, idx]
quats_d   = quats[:, idx]
ang_vel_d = ang_vel[:, idx]

df = pd.DataFrame({
    'timestamp':  t,
    'px': pos[0],    'py': pos[1],    'pz': pos[2],
    'qw': quats[0],  'qx': quats[1],  'qy': quats[2],  'qz': quats[3],
    'vx': vel[0],    'vy': vel[1],    'vz': vel[2],
    'wx': ang_vel[0],'wy': ang_vel[1],'wz': ang_vel[2],
})
df_discrete = df.iloc[idx]
df_discrete.to_csv('trajectory_1hz.csv', index=False)




# --- Plots ---
fig = plt.figure(figsize=(14, 9))

# 1. Top-down view
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(pos_d[0], pos_d[1], color='k', zorder=5, s=30, label='1Hz samples')
rect = plt.Polygon([[-3,-3],[3,-3],[3,3],[-3,3]], fill=False, edgecolor='r', linewidth=2)
ax1.add_patch(rect)
ax1.plot(px, py, 'b', linewidth=1.5)
ax1.plot(0, 0, 'r*', markersize=12, label='Target')
ax1.plot(px[0], py[0], 'go', markersize=8, label='Start')
ax1.plot(px[-1], py[-1], 'rs', markersize=8, label='End')
ax1.set_xlim(-3.5, 3.5); ax1.set_ylim(-3.5, 3.5)
ax1.set_aspect('equal'); ax1.grid(True)
ax1.set_title('Top View - 6x6m Testbed')
ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
ax1.legend(fontsize=8)

# 2. 3D trajectory
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot(px, py, pz, 'b', linewidth=1.5)
ax2.scatter(0, 0, 0, color='r', s=80, marker='*')
ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_zlabel('Z (m)')
ax2.set_title('3D Trajectory')

# 3. Quaternion
ax3 = fig.add_subplot(2, 3, 3)
for val, label in zip(quats, ['w','x','y','z']):
    ax3.plot(t, val, label=label)
ax3.set_title('Quaternion'); ax3.set_xlabel('t (s)')
ax3.legend(); ax3.grid(True)

# 4. Linear velocity
ax4 = fig.add_subplot(2, 3, 4)
for val, label in zip(vel, ['vx','vy','vz']):
    ax4.plot(t, val, label=label)
ax4.set_title('Linear Velocity (m/s)'); ax4.set_xlabel('t (s)')
ax4.legend(); ax4.grid(True)

# 5. Angular velocity
ax5 = fig.add_subplot(2, 3, 5)
for val, label in zip(ang_vel, ['wx','wy','wz']):
    ax5.plot(t, val, label=label)
ax5.set_title('Angular Velocity (rad/s)'); ax5.set_xlabel('t (s)')
ax5.legend(); ax5.grid(True)

# 6. Radius over time
ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(t, np.sqrt(px**2 + py**2), 'b')
ax6.axhline(r_max, color='r', linestyle='--', label='Testbed limit')
ax6.set_title('Radius over Time'); ax6.set_xlabel('t (s)'); ax6.set_ylabel('r (m)')
ax6.legend(); ax6.grid(True)

plt.tight_layout()
plt.show()