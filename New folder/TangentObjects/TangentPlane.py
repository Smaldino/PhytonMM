import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create figure with 3 subplots
fig = plt.figure(figsize=(18, 6))

# ============================================================================
# 1. TANGENT LINE (2D) - f(x) = x^2 at x0 = 1
# ============================================================================
ax1 = fig.add_subplot(131)
x = np.linspace(-0.5, 2.5, 100)
f_1d = x**2

# Point of tangency
x0 = 1.0
f0 = x0**2
df_dx = 2*x0  # derivative

# Tangent line
L = f0 + df_dx * (x - x0)

# Plot
ax1.plot(x, f_1d, 'b-', linewidth=2.5, label='$f(x) = x^2$')
ax1.plot(x, L, 'r--', linewidth=2, label=f'Tangent line at $x_0={x0}$')
ax1.plot(x0, f0, 'ko', markersize=10, label=f'Point $P=({x0}, {f0})$')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Tangent Line to $f(x)=x^2$ at $x_0=1$', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_aspect('equal', adjustable='box')

# ============================================================================
# 2. TANGENT PLANE (3D) - f(x,y) = x^2 + y^2 at (0.5, 0.5)
# ============================================================================
ax2 = fig.add_subplot(132, projection='3d')

# Create grid
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # Surface

# Point of tangency
x0, y0 = 0.5, 0.5
z0 = x0**2 + y0**2
df_dx = 2*x0
df_dy = 2*y0

# Tangent plane
Z_plane = z0 + df_dx*(X - x0) + df_dy*(Y - y0)

# Plot surface
surf = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)

# Plot tangent plane
ax2.plot_surface(X, Y, Z_plane, color='red', alpha=0.4, linewidth=0)

# Point of tangency
ax2.scatter([x0], [y0], [z0], color='black', s=100, label=f'Point $P=({x0}, {y0}, {z0:.2f})$')

# Normal vector (for visualization)
normal = np.array([-df_dx, -df_dy, 1])
normal_unit = normal / np.linalg.norm(normal)
ax2.quiver(x0, y0, z0, normal_unit[0], normal_unit[1], normal_unit[2], 
           length=1.2, color='blue', linewidth=2.5, label='Normal vector')

ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y', fontsize=11)
ax2.set_zlabel('z', fontsize=11)
ax2.set_title('Tangent Plane to $f(x,y)=x^2+y^2$ at $(0.5,0.5)$', 
              fontsize=14, fontweight='bold', pad=20)
ax2.view_init(elev=25, azim=-60)
ax2.legend(loc='upper left', fontsize=9)

# ============================================================================
# 3. TANGENT HYPERPLANE CONCEPT (4D visualization via level sets)
# Since we can't visualize 4D directly, we show level sets of f(x,y,z) = x^2+y^2+z^2
# and illustrate the tangent plane to a level surface in 3D (analogous concept)
# ============================================================================
ax3 = fig.add_subplot(133, projection='3d')

# Create sphere (level surface of f(x,y,z)=x^2+y^2+z^2 = r^2)
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
r = 1.0
X_sph = r * np.outer(np.cos(u), np.sin(v))
Y_sph = r * np.outer(np.sin(u), np.sin(v))
Z_sph = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Point on sphere
theta0, phi0 = np.pi/4, np.pi/4
x0_sph = r * np.cos(theta0) * np.sin(phi0)
y0_sph = r * np.sin(theta0) * np.sin(phi0)
z0_sph = r * np.cos(phi0)

# Normal vector to sphere (gradient of F=x^2+y^2+z^2)
normal_sph = np.array([x0_sph, y0_sph, z0_sph])
normal_sph_unit = normal_sph / np.linalg.norm(normal_sph)

# Tangent plane to sphere at P: normal · (X - P) = 0
# We'll plot a small patch of the tangent plane
plane_size = 0.6
xx = np.linspace(-plane_size, plane_size, 10)
yy = np.linspace(-plane_size, plane_size, 10)
XX, YY = np.meshgrid(xx, yy)

# Create orthonormal basis for tangent plane
t1 = np.array([1, 0, 0]) - np.dot([1,0,0], normal_sph_unit)*normal_sph_unit
t1 = t1 / np.linalg.norm(t1)
t2 = np.cross(normal_sph_unit, t1)

# Parametrize tangent plane patch
X_plane = x0_sph + XX*t1[0] + YY*t2[0]
Y_plane = y0_sph + XX*t1[1] + YY*t2[1]
Z_plane = z0_sph + XX*t1[2] + YY*t2[2]

# Plot sphere
ax3.plot_surface(X_sph, Y_sph, Z_sph, color='cyan', alpha=0.3, linewidth=0)

# Plot tangent plane patch
ax3.plot_surface(X_plane, Y_plane, Z_plane, color='red', alpha=0.6, linewidth=0)

# Point of tangency
ax3.scatter([x0_sph], [y0_sph], [z0_sph], color='black', s=100, 
            label=f'Point $P=({x0_sph:.2f}, {y0_sph:.2f}, {z0_sph:.2f})$')

# Normal vector
ax3.quiver(x0_sph, y0_sph, z0_sph, 
           normal_sph_unit[0], normal_sph_unit[1], normal_sph_unit[2], 
           length=1.0, color='blue', linewidth=2.5, label='Normal vector')

ax3.set_xlabel('x', fontsize=11)
ax3.set_ylabel('y', fontsize=11)
ax3.set_zlabel('z', fontsize=11)
ax3.set_title('Tangent Plane to Sphere\n(Analogous to Tangent Hyperplane)', 
              fontsize=14, fontweight='bold', pad=15)
ax3.view_init(elev=20, azim=30)
ax3.legend(loc='upper left', fontsize=9)
ax3.set_box_aspect([1,1,1])

# Final layout
plt.tight_layout()
plt.savefig('tangent_objects.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# BONUS: 2D contour view showing tangent line approximation quality
# ============================================================================
fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Zoomed view near tangency point
x_zoom = np.linspace(0.7, 1.3, 100)
f_zoom = x_zoom**2
L_zoom = 1 + 2*(x_zoom - 1)

ax4.plot(x_zoom, f_zoom, 'b-', linewidth=2.5, label='$f(x)=x^2$')
ax4.plot(x_zoom, L_zoom, 'r--', linewidth=2, label='Tangent line')
ax4.fill_between(x_zoom, f_zoom, L_zoom, alpha=0.3, color='gray', 
                  label=f'Error ≈ ${(1.3-1)**2:.3f}$ at edge')
ax4.plot(1, 1, 'ko', markersize=10)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.set_title('Zoomed View: Excellent Approximation Near $x_0$', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.set_aspect('equal')

# Right: Error visualization
error = np.abs(f_zoom - L_zoom)
ax5.plot(x_zoom, error, 'g-', linewidth=2.5)
ax5.axhline(y=0, color='k', linewidth=0.5)
ax5.fill_between(x_zoom, 0, error, alpha=0.3, color='green')
ax5.set_xlabel('x', fontsize=12)
ax5.set_ylabel('Approximation Error', fontsize=12)
ax5.set_title('Error Grows Quadratically with Distance', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.text(1.0, 0.02, '$\\text{Error} \\approx (x-x_0)^2$', 
         fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('tangent_error.png', dpi=150, bbox_inches='tight')
plt.show()