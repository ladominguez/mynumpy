import numpy as np
import math
import matplotlib.pyplot as plt

# Define function to calculate Taylor operators
def _central_difference_coefficients(nop, n):
    """
    Calculate the central finite difference stencil for an arbitrary number
    of points and an arbitrary order derivative.
    
    :param nop: The number of points for the stencil. Must be
        an odd number.
    :param n: The derivative order. Must be a positive number.
    """
    m = np.zeros((nop, nop))
    
    for i in range(nop):
        for j in range(nop):
            dx = j - nop // 2
            m[i, j] = dx ** i
    
    s = np.zeros(nop)
    s[n] = math.factorial(n)
    
    # The following statement return oper = inv(m) s
    oper = np.linalg.solve(m, s)
    
    # Calculate operator
    return oper


def derivative(t, y, nop=3, n=1):
    """
    Estimate the derivative using n points.

     param: y function
     param: dt increment
     param: nop number of points
     param: n order of the derivative
    """

    dt = t[2] - t[1]
    oper = _central_difference_coefficients(nop, n)
    oper = np.flip(oper)

    y_derivative = np.convolve(y,oper,'same')/dt**n
    nn = nop//2
    y_derivative = y_derivative[nn:-nn]
    t_derivative = t[nn:-nn]
    return t_derivative, y_derivative

if __name__ == '__main__':
    t = np.linspace(0,1,101)
    
    y = np.sin(2*np.pi*t)
    tp, yp = derivative(t, y, nop=10)
    ya = 2*np.pi*np.cos(2*np.pi*tp)
    plt.plot(t,y)
    plt.plot(tp, yp, 'ko')
    plt.plot(tp,ya)
    plt.show()

    
