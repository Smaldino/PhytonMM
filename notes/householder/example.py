import numpy as np

# Original and Target
x = np.array([3, 4])
y = np.array([5, 0]) # Target on x-axis (length is 5)

# The 'Normal' to the mirror
u = x - y 

# The 'Reflection' calculation broken down
proj_x_onto_u = (np.dot(u, x) / np.dot(u, u)) * u

step_0 = x
step_1 = x - proj_x_onto_u        # Landing on the mirror
step_2 = x - 2 * proj_x_onto_u    # Reflecting to y

print(f"Step 0 (Start): {step_0}")
print(f"Step 1 (On Mirror): {step_1}")
print(f"Step 2 (Target y): {step_2}")