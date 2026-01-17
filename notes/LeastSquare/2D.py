import numpy as np
import matplotlib.pyplot as plt

# Define direction vector (column of A)
a = np.array([2, 1])
b = np.array([3, 4])

# Projection of b onto a:
# proj = (a·b / a·a) * a
proj = np.dot(a, b) / np.dot(a, a) * a
residual = b - proj

# Plot
plt.figure(figsize=(6,6))
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)

# Draw the line (column space)
t = np.linspace(-1, 3, 100)
line_points = np.outer(t, a)
plt.plot(line_points[:,0], line_points[:,1], 'b-', label='Column space (line)')

# Vectors
plt.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='red', label='b (data)')
plt.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, color='green', label='Projection')
plt.quiver(proj[0], proj[1], residual[0], residual[1], angles='xy', scale_units='xy', scale=1, color='orange', label='Residual')

plt.scatter(b[0], b[1], color='red')
plt.scatter(proj[0], proj[1], color='green')

plt.legend()
plt.grid(True)
plt.axis('equal')
plt.title('Projection of b onto Column Space (2D)')
plt.xlabel('x'); plt.ylabel('y')
plt.show()