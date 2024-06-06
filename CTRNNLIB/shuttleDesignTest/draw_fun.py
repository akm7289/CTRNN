import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return 3*x**4 + 5*x**2 + 7*x + 8
# Generate x values
x = np.linspace(-2, 2, 400)  # Adjust the range and number of points as needed

# Calculate corresponding y values
y = f(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = 3x^4 + 5x^2 + 7x + 8', color='b')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) = 3x^4 + 5x^2 + 7x + 8')
plt.legend()

plt.show()
