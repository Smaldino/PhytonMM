import numpy as np
import matplotlib.pyplot as plt

def orthogonal_projection_demo():
    # 1. Setup the direction vector u
    # In your notes, P = uu^T assumes u is a unit vector (norm = 1)
    u_raw = np.array([2, 1]) 
    u = u_raw / np.linalg.norm(u_raw)
    
    # 2. Setup the vector x to be projected
    x = np.array([1, 3])
    
    # 3. Create the Projection Matrix P ("The Shadow Maker")
    # P = u * u.T
    P = np.outer(u, u)
    
    # 4. Create the Complementary Projector Q ("The Rejection Matrix")
    # Q = I - P
    I = np.eye(2)
    Q = I - P
    
    # 5. Calculate the components
    x_parallel = P @ x  # Shadow of x on the line spanned by u
    x_perp = Q @ x      # The "lifting vector" from the line up to x
    
    # --- Output results ---
    print(f"Original x: {x}")
    print(f"x_parallel (Shadow): {x_parallel}")
    print(f"x_perp (Lifting vector): {x_perp}")
    print(f"Verification (x_para + x_perp = x): {x_parallel + x_perp}")
    
    # --- Visualization ---
    plt.figure(figsize=(8, 8))
    origin = np.array([0, 0])
    
    # Draw the line spanned by u
    line_scale = 4
    plt.plot([-u[0]*line_scale, u[0]*line_scale], [-u[1]*line_scale, u[1]*line_scale], 
             color='black', linestyle='--', alpha=0.3, label='Line spanned by u')

    # Draw the Vectors
    plt.quiver(*origin, x[0], x[1], color='red', angles='xy', scale_units='xy', scale=1, label='x (Original)')
    plt.quiver(*origin, x_parallel[0], x_parallel[1], color='blue', angles='xy', scale_units='xy', scale=1, label='x∥ (Projected Shadow)')
    plt.quiver(x_parallel[0], x_parallel[1], x_perp[0], x_perp[1], color='green', angles='xy', scale_units='xy', scale=1, label='x⊥ (Orthogonal Rejection)')

    # Formatting
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.gca().set_aspect('equal')
    plt.title("Orthogonal Projection: x = Px + Qx")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    orthogonal_projection_demo()