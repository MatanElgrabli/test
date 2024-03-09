import numpy as np
import matplotlib.pyplot as plt

L = 25
c = 3


# Define the piecewise function
def f(x):
    # Define the conditions
    condition1 = (x - L / 2 <= 0)
    condition2 = (x - L / 2 > 0)

    # Define the functions for each condition
    func1 = lambda x: (2 * x / L) * c
    func2 = lambda x: (L - x) * 2 * c / L

    # Apply the functions based on the conditions
    result = np.piecewise(x, [condition1, condition2],
                          [func1, func2])

    return result


# Generate an array of x values
x = np.linspace(0, L, 1000)

# Apply the piecewise function
y = f(x)

# Plot the result

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Piecewise Function')
plt.grid(True)
plt.show()

## %%
import numpy as np

def fourier_coefficients(func, n, L):
    """
    Calculate the Fourier coefficients of a function up to the nth term for the sine basis functions.

    Parameters:
        func (callable): The function for which Fourier coefficients are calculated.
        n (int): The number of Fourier coefficients to calculate.
        L (float): The period of the function.

    Returns:
        ndarray: Array containing the Fourier coefficients up to the nth term for the sine basis functions.
    """
    b_coeffs = np.zeros(n)  # Coefficients for sine terms

    # Define the basis function for sine terms
    def sin_basis(x, k):
        return np.sin( np.pi * k * x / L)

    # Calculate the coefficients
    for k in range(1, n + 1):
        b_coeffs[k - 1] = (2 / L) * np.trapz(func(x) * sin_basis(x, k), x)

    return b_coeffs

# Calculate the Fourier coefficients for the sine basis functions
n = 100  # Number of Fourier coefficients to calculate
b_coeffs = fourier_coefficients(f, n, L)

# Print the Fourier coefficients
print("Fourier coefficients (b_k):", b_coeffs)

def fourier_series(x, coeffs, L):
    """
    Calculate the Fourier series given the x values, Fourier coefficients, and period.

    Parameters:
        x (ndarray): Array of x values.
        coeffs (ndarray): Array containing the Fourier coefficients.
        L (float): The period of the function.

    Returns:
        ndarray: Array containing the Fourier series values corresponding to the x values.
    """
    series = np.zeros_like(x)
    for k, coeff in enumerate(coeffs, start=1):
        series += coeff * np.sin( np.pi * k * x / L)
    return series


plt.figure(figsize=(8, 6))
plt.plot(x, fourier_series(x, b_coeffs, L), label='Fourier series', color='red')  # Fourier series
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fourier Series')
plt.legend()
plt.grid(True)
plt.show()

# Values of n
n_values = [1, 2, 3, 7, 30,100]

# Create subplots
plt.figure(figsize=(10, 20))
for i, n in enumerate(n_values, start=1):
    # Calculate the Fourier coefficients for the sine basis functions
    b_coeffs = fourier_coefficients(f, n, L)

    # Calculate the Fourier series
    series = fourier_series(x, b_coeffs, L)

    # Plot the Fourier series
    plt.subplot(len(n_values), 1, i)
    plt.plot(x, f(x), label='Original function', linestyle='--', color='blue')  # Original function
    plt.plot(x, series, label=f'Fourier series (n={n})', color='red')  # Fourier series
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Fourier Series (n={n})')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

## time dependent solution
v = 0.5 # wave velocity
def psi(x,t, coeffs, L):

    series = np.zeros_like(x)
    for n, coeff in enumerate(coeffs, start=1):
        k_n = np.pi * n/ L
        series += coeff * np.cos(v*k_n*t)* np.sin( k_n * x )
    return series

t_values = [0.5,1, 2, 5,10]

# Create subplots
plt.figure(figsize=(10, 20))
for i, t in enumerate(t_values, start=1):
    # Calculate the Fourier coefficients for the sine basis functions
    b_coeffs = fourier_coefficients(f,100 , L)

    # Calculate the Fourier series
    series = psi(x,t, b_coeffs, L)

    # Plot the Fourier series
    plt.subplot(len(t_values), 1, i)
    plt.plot(x, f(x), label='Original function', linestyle='--', color='blue')  # Original function
    plt.plot(x, series, label=f'Fourier series (t={t})', color='red')  # Fourier series
    plt.xlabel('x')
    plt.ylabel('psi')
    plt.title(f'Fourier Series (t={t})')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
