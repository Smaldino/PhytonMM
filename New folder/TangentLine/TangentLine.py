import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 200)
plt.plot(x, np.sin(x), 'b', label='sin(x)')
plt.plot(x, x, 'r--', label='Tangent line y=x at x=0')
plt.axvline(0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
plt.legend(); plt.title('Tangent Line Approximation'); plt.grid(True)
plt.show()