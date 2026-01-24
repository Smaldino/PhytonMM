import matplotlib.pyplot as plt

z = 3 + 4j
plt.figure(figsize=(5,5))
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.scatter(z.real, z.imag, color='red', s=100)
plt.arrow(0, 0, z.real, z.imag, head_width=0.2, color='blue')
plt.xlabel('Re(z)'); plt.ylabel('Im(z)')
plt.title('Complex Number in the Plane')
plt.grid(True); plt.axis('equal')
plt.show()