import numpy as np
import matplotlib.pyplot as plt

def plot_rotation(theta_deg):
    theta = np.radians(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    v = np.array([2, 1])          # Original vector
    v_rot = R @ v                 # Rotated vector
    
    plt.figure(figsize=(6, 6))
    plt.arrow(0, 0, v[0], v[1], head_width=0.15, color='blue', 
              label=f'Original v = {v}')
    plt.arrow(0, 0, v_rot[0], v_rot[1], head_width=0.15, color='red',
              label=f'Rotated v\' = {v_rot.round(2)}')
    
    # Draw arc showing rotation angle
    arc = np.linspace(0, theta, 50)
    plt.plot(0.5*np.cos(arc), 0.5*np.sin(arc), 'k--', lw=1)
    plt.text(0.3*np.cos(theta/2), 0.3*np.sin(theta/2), f'{theta_deg}°', fontsize=12)
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'2D Rotation by {theta_deg}°')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()
    
    print(f"Rotation matrix R({theta_deg}°):\n{R.round(3)}")
    print(f"||v|| = {np.linalg.norm(v):.3f}, ||Rv|| = {np.linalg.norm(v_rot):.3f} ✓")

plot_rotation(45)