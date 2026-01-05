import numpy as np
import matplotlib.pyplot as plt

def plot_householder_mirror():
    # 1. Setup the vectors
    x = np.array([3, 4])
    # Target y has the same length as x but lies on the x-axis
    norm_x = np.linalg.norm(x)
    y = np.array([norm_x, 0])
    
    # 2. Define the Normal and the Mirror
    u = x - y  # Vector normal to the mirror
    # To draw the mirror, we find a vector perpendicular to u
    mirror_v = np.array([-u[1], u[0]]) 
    
    # 3. Create the Plot
    plt.figure(figsize=(8, 8))
    origin = np.array([0, 0])

    # Draw the Mirror (Hyperplane)
    m_ext = 1.5
    plt.plot([-mirror_v[0]*m_ext, mirror_v[0]*m_ext], 
             [-mirror_v[1]*m_ext, mirror_v[1]*m_ext], 
             color='black', linestyle='--', linewidth=2, label='The Mirror (Hyperplane)')

    # Draw Vector x (The Object)
    plt.quiver(*origin, x[0], x[1], color='red', angles='xy', scale_units='xy', scale=1, 
                width=0.015, label='x (Original)')

    # Draw Vector y (The Reflection)
    plt.quiver(*origin, y[0], y[1], color='blue', angles='xy', scale_units='xy', scale=1, 
                width=0.015, label='y (Reflected)')

    # Draw the Path (u vector) connecting x and y
    plt.quiver(y[0], y[1], u[0], u[1], color='green', angles='xy', scale_units='xy', scale=1, 
                width=0.01, alpha=0.5, label='u = x - y (Normal)')

    # Fill the "light" area to show the reflection symmetry
    poly_points = np.array([origin, x, y])
    plt.fill(poly_points[:,0], poly_points[:,1], 'yellow', alpha=0.1)

    # Annotations for clarity
    plt.text(x[0], x[1], '  x', color='red', fontsize=12, fontweight='bold')
    plt.text(y[0], y[1], '  y', color='blue', fontsize=12, fontweight='bold')
    
    # Plot Limits and Grid
    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal')
    plt.title("Python Graph: The Householder Mirror Analogy")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.show()

if __name__ == "__main__":
    plot_householder_mirror()