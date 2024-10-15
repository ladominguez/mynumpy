import numpy as np
import math
import matplotlib.pyplot as plt


def fLOESS(x,y, span):
    """
    Performs LOESS (Locally Weighted Scatterplot Smoothing) with a 2nd order polynomial.
    
    Parameters:
    noisy (array): Either a 1D array of noisy data or a 2D array where the first column is x-data 
                   and the second column is noisy y-data.
    span (float): A value specifying the fraction of data to use for smoothing. Minimum is 4/n.
    
    Returns:
    smoothed (array): Smoothed y-data.
    """
    
    # Error checking
    if len(noisy) * span < 4:
        raise ValueError('The input "span" is too low')

    # Check if noisy is a 1D array, if so create default x data
    if noisy.ndim == 1:
        noisy = np.column_stack((np.arange(1, len(noisy)+1), noisy))

    # Define variables
    #x = noisy[:, 0]
    #y = noisy[:, 1]
    n = len(x)
    r = x[-1] - x[0]
    
    hlims = np.array([[span, x[0]],
                      [span / 2, x[0] + r * span / 2],
                      [span / 2, x[0] + r * (1 - span / 2)],
                      [span, x[-1]]])

    smoothed = np.zeros(n)

    for i in range(n):
        # Define the tricube weight function
        h = np.interp(x[i], hlims[:, 1], hlims[:, 0])
        w = (1 - np.abs((x / max(x) - x[i] / max(x)) / h) ** 3) ** 3

        # Filter the points that fall within the span (for speed)
        w_idx = w > 0
        w_ = w[w_idx]
        x_ = x[w_idx]
        y_ = y[w_idx]

        # Weighted polynomial regression coefficients
        XX = np.array([[np.nansum(w_ * x_ ** 0), np.nansum(w_ * x_ ** 1), np.nansum(w_ * x_ ** 2)],
                       [np.nansum(w_ * x_ ** 1), np.nansum(w_ * x_ ** 2), np.nansum(w_ * x_ ** 3)],
                       [np.nansum(w_ * x_ ** 2), np.nansum(w_ * x_ ** 3), np.nansum(w_ * x_ ** 4)]])

        YY = np.array([np.nansum(w_ * y_ * (x_ ** 0)),
                       np.nansum(w_ * y_ * (x_ ** 1)),
                       np.nansum(w_ * y_ * (x_ ** 2))])

        # Solve for coefficients
        CC = np.linalg.solve(XX, YY)

        # Calculate the fitted value for the current point
        smoothed[i] = CC[0] + CC[1] * x[i] + CC[2] * x[i] ** 2

    return smoothed


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

    # TODO Convert to unit test
    #t = np.linspace(0,1,101)
    #y = np.sin(2*np.pi*t)
    #tp, yp = derivative(t, y, nop=10)
    #ya = 2*np.pi*np.cos(2*np.pi*tp)
    #plt.plot(t,y)
    #plt.plot(tp, yp, 'ko')
    #plt.plot(tp,ya)
    #plt.show()
    #
    # 
    #
    # 
    # 
    x = 10 * np.sort(np.random.rand(100, 1), axis=0)
    clean = np.cos(1.0 * x + 1) + 1.0
    noisy = np.hstack((x, clean + 1.5 * (np.random.rand(len(x), 1) - 0.5)))

    # Define span length (randomized between 10 and length of data)
    span = 0.1 + 0.9 * np.random.rand(1, 1)[0][0]

    # Fit the data
    smoothed = fLOESS(noisy, span)

    # Plot the data
    plt.figure()
    plt.plot(x, clean, 'k', label='clean')
    plt.plot(x, noisy[:, 1], '.', label='noisy')
    plt.plot(x, smoothed, linewidth=2, label=f'smoothed (span = {span:.2f})')
    plt.legend()
    plt.show()
    pass


    
