import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define two basis vectors (columns of A)
a1 = np.array([1, 0, 1])
a2 = np.array([0, 1, 1])
A = np.column_stack([a1, a2])

# True b (not in column space)
b = np.array([1, 2, 1])

# Solve least squares
x_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
proj = A @ x_ls  # projection of b onto col(A)
residual = b - proj

# Plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Column space (plane spanned by a1, a2)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 10), np.linspace(-0.5, 2.5, 10))
zz = xx + yy  # since both a1 and a2 have z=1
ax.plot_surface(xx, yy, zz, alpha=0.2, color='lightblue')

# Vectors
origin = np.zeros(3)
ax.quiver(*origin, *a1, color='red', label='a1')
ax.quiver(*origin, *a2, color='green', label='a2')
ax.quiver(*origin, *b, color='black', label='b (data)')
ax.quiver(*origin, *proj, color='purple', label='Projection')
ax.quiver(*proj, *residual, color='orange', label='Residual', arrow_length_ratio=0.1)

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.title('Geometric View: Projection onto Column Space')
plt.show()