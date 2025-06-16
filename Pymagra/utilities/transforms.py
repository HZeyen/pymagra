
# -*- coding: utf-8 -*-
"""
Last modified on June 15, 2025

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         Université Paris-Saclay, France

Contains the following functions:
    - matrix_extension : Extraplote data to fill a full rectangle
    - pole_reduction : Reduce magnetic data to the pole
    - continuation : Continue data to another measurement height
    - horizontal_derivative2 : Calculates second horizontal derivative using
      finite differences. Doesn't seem to be used anywhere.
    - horizontal_derivative : Calculate absolute horizontal derivative using
      finite differences
    - vertical_derivative2 : Calculate second vertical derivative using finite
      differences
    - vertical_derivative : Calculate vertical derivative using FFT
    - analytic_signal : Calculate analytic signal of 2D data
    - tilt : Calculate tilt angle
    - log_spect : Calculate logarithmic power spectrum using FFT
    - spector_line : Estimate depths from power spectrum dacay of one line
    - spector1D : Estimate depths from power spectrum dacay of all lines
    - spector2D : Estimate source depth using spectral decay using sliding
      windows
    - gradient : Calculate smoothed gradient of 2D gridded data

"""

import numpy as np
from PyQt5 import QtWidgets
from . import utilities as utils
from ..in_out import communication as comm


def matrix_extension(data):
    """
    Creation of extended matix for 2D Fourier transform.
    The toutine mirrors the lower half of the matrix and adds it at the
    bottom and mirrors the upper half to the top. Equivalently right and
    left

    Parameters
    ----------
    data : 2D numpy array
        Data matrix to be extended

    Returns
    -------
    d : 2D numpy array extended in both directions
        (ny1,nx1): Tuple with starting indices of the original data in
        matrix d
        (ny2,nx2): Tuple with final indices of the original data in matrix
        d plus one.
        The original data may thus be retrieved as
        data = d[ny1:ny2,nx1:nx2]

    """
    ny, nx = data.shape
    d = np.zeros((2 * (ny - 1), 2 * (nx - 1)))
    nx_add_left = int((nx - 2) / 2)
    nx_add_right = int((nx - 1) / 2)
    ix_add_left = 1 + nx_add_left
    ix_add_right = nx - nx_add_right - 1

    ny_add_down = int((ny - 2) / 2)
    ny_add_up = int((ny - 1) / 2)
    iy_add_down = 1 + ny_add_down
    iy_add_up = d.shape[0] - ny

    nx_right = nx_add_left + nx
    ny_up = ny_add_down + ny
    d[ny_add_down:ny_up, nx_add_left:nx_right] = data
    d[ny_add_down:ny_up, 0:nx_add_left] = np.flip(data[:, 1:ix_add_left],
                                                  axis=1)
    d[ny_add_down:ny_up, nx_right:] = np.flip(data[:, ix_add_right: nx-1],
                                              axis=1)
    d[:ny_add_down, :] = np.flip(d[iy_add_down: iy_add_down+ny_add_down, :],
                                 axis=0)
    d[ny_up:, :] = np.flip(d[iy_add_up: iy_add_up+ny_add_up, :], axis=0)
    return d, (ny_add_down, nx_add_left), (ny_add_down+ny, nx_add_left+nx)


def pole_reduction(data, dx, dy, inc, dec):
    """
    Calculation of pole-reduced magnetic data supposing only induced
    magnetization.
    Formula from Keating and Zerbo, Geophysics 61, nᵒ 1 (1996): 131‑137.


    Parameters
    ----------
    data : 2D numpy float array
        Original magnetic data interpolated on a regular grid which may
        have different grid width in x (E-W) and y (N-S) direction.
    dx : float
        grid step in x direction.
    dy : float
        grid step in y direction.
    inc : float
        Inclination of magnetic field [degrees].
    gec : float
        Declination of magnetic field [degrees].

    Returns
    -------
    d : 2D numpy float array with the same shape as data
        Reduced to the pole magnetic data

    """
    fac = np.pi / 180.0
    i = inc * fac
    d = dec * fac
    cz = np.sin(i)
    cI = np.cos(i)
    sD = np.sin(d)
    cD = np.cos(d)
    cy = cI * cD
    cx = cI * sD
    d, corner1, corner2 = matrix_extension(data)
    ny1 = corner1[0]
    nx1 = corner1[1]
    ny2 = corner2[0]
    nx2 = corner2[1]
    dF = np.fft.fft2(d)
    ny, nx = d.shape
    kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
    u = np.outer(np.ones_like(ky), kx)
    v = np.outer(ky, np.ones_like(kx))
    s = np.sqrt(u**2 + v**2)
    s[0, 0] = 1.0
    fac = (1j * (cx * u + cy * v) / s + cz) ** 2
    fac[0, 0] = 1.0
    dF /= fac
    d = np.fft.ifft2(dF)
    return np.real(d[ny1:ny2, nx1:nx2])


def continuation(data, dx, dy, dz):
    """
    Vertical continuation of potential field data using Fourier transform

    Parameters
    ----------
    data : 2D numpy float array
        Data interpolated onto a regular grid
    dx, dy : float
        Grid spacing in x and y direction [m]
    dz : float
        Distance to continue data [m], positive upwards

    Returns
    -------
    2D numpy float array, same shape as data
        Prolongated data

    """
    d, corner1, corner2 = matrix_extension(data)
    ny1 = corner1[0]
    nx1 = corner1[1]
    ny2 = corner2[0]
    nx2 = corner2[1]
    dF = np.fft.fft2(d)
    ny, nx = d.shape
    kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
    u = np.outer(np.ones_like(ky), kx)
    v = np.outer(ky, np.ones_like(kx))
    s = np.sqrt(u**2 + v**2)
    dF *= np.exp(-s * dz)
    d = np.fft.ifft2(dF)
    return np.real(d[ny1:ny2, nx1:nx2])


def horizontal_derivative2(data, dx, dy):
    """
    Second horizontal derivative of potential field data using finite
    differences

    Parameters
    ----------
    data : 2D numpy float array
           Data interpolated onto a regular grid
    dx, dy : float
           Grid spacing in x and y direction [m]

    Returns
    -------
    2D numpy float array, same shape as data
        2nd horizontal derivative of data
    """
    gx = np.zeros_like(data)
    gx[:, 1:-1] = (data[:, 2:] + data[:, :-2] - 2 * data[:, 1:-1]) / dx**2
    gx[:, 0] = 2 * gx[:, 1] - gx[:, 2]
    gx[:, -1] = 2 * gx[:, -1] - gx[:, -2]
    gy = np.zeros_like(data)
    gy[1:-1, :] = (data[2:, :] + data[:-2, :] - 2 * data[1:-1, :]) / dy**2
    gy[0, :] = 2 * gy[1, :] - gy[2, :]
    gy[-1, :] = 2 * gy[-1, :] - gy[-2, :]
    return np.sqrt(gx**2 + gy**2)


def horizontal_derivative(data, dx, dy):
    """
    First horizontal derivative of potential field data using finite
    differences

    Parameters
    ----------
    data : 2D numpy float array
           Data interpolated onto a regular grid
    dx, dy : float
           Grid spacing in x and y direction [m]

    Returns
    -------
    2D numpy float array, same shape as data
        First horizontal derivative of data
    """
    gx = np.zeros_like(data)
    gx[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (dx * 2)
    gx[:, 0] = (data[:, 1] - data[:, 0]) / dx
    gx[:, -1] = (data[:, -1] - data[:, -2]) / dx
    gy = np.zeros_like(data)
    gy[1:-1, :] = (data[2:, :] - data[:-2, :]) / (2 * dy)
    gy[0, :] = (data[1, :] - data[0, :]) / dy
    gy[-1, :] = (data[-1, :] - data[-2, :]) / dy
    return np.sqrt(gx**2 + gy**2)


def vertical_derivative2(data):
    """
    Second vertical derivative of potential field data using finite
    differences

    Parameters
    ----------
    data : 2D numpy float array
           Data interpolated onto a regular grid

    Returns
    -------
    2D numpy float array, same shape as data
        2nd vertical derivative of data
    """
    gz2 = np.zeros_like(data)
    gz2[:, :] = np.nan
    gz2[1:-1, 1:-1] = (data[:-2, 1:-1] + data[2:, 1:-1] + data[1:1, :-2]
                       + data[1:-1, 2:] - 4 * data[1:-1, 1:-1])
    return gz2


def vertical_derivative(data, dx, dy):
    """
    First vertical derivative of potential field data using Fourier
    transform

    Parameters
    ----------
    data : 2D numpy float array
           Data interpolated onto a regular grid
    dx, dy : float
           Grid spacing in x and y direction [m]

    Returns
    -------
    2D numpy float array, same shape as data
        First vertical derivative of data
    """
    d, corner1, corner2 = matrix_extension(data)
    dF = np.fft.fft2(d)
    ny1 = corner1[0]
    nx1 = corner1[1]
    ny2 = corner2[0]
    nx2 = corner2[1]
    ny, nx = d.shape
    kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
    u = np.outer(np.ones_like(ky), kx)
    v = np.outer(ky, np.ones_like(kx))
    s = np.sqrt(u**2 + v**2)
    dF *= s
    return np.real(np.fft.ifft2(dF)[ny1:ny2, nx1:nx2])


def analytic_signal(data, dx, dy):
    """
    Calculation of analytical signal of potential field data via
    vertical and horizontal derivatives

    Parameters
    ----------
    data : 2D numpy float array
           Data interpolated onto a regular grid
    dx, dy : float
           Grid spacing in x and y direction [m]

    Returns
    -------
    2D numpy float array, same shape as data
        Analytic signal
    """
    gh = horizontal_derivative(data, dx, dy)
    gv = vertical_derivative(data, dx, dy)
    return np.sqrt(gh**2 + gv**2)


def tilt(data, dx, dy, grad=None):
    """
    Tilt angle of potential field data using Fourier transform

    Parameters
    ----------
    data : 2D numpy float array
           Data interpolated onto a regular grid
    dx, dy : float
           Grid spacing in x and y direction [m]
    grad : 2D numpy float array, optional; Default: None
           Vertical derivative of data if it has been measured
           If None, it is calculated numerically using FFT.

    Returns
    -------
    2D numpy float array, same shape as data
        Tilt angle of data
    """
    if grad is None:
        grad = vertical_derivative(data, dx, dy)
    grad2 = vertical_derivative(grad, dx, dy)
    gh = horizontal_derivative(data, dx, dy)
    return np.arctan2(grad, abs(gh)), grad, grad2, gh


def log_spect(data, d, n_coef):
    """
    Calculate logarithmic power spectrum of a series of data

    Parameters
    ----------
    data : numpy 1D array, float
        Data to be analyzed.
    d : float
        Distance between data points (sampling distance).
    n_coef : int
        Number of coefficients of the spectrum to be returned.

    Returns
    -------
    numpy 1D array
        logarithm of normalized power spectrum.
    list
        Wavenumbers of power spectrum.

    """
    index = np.isfinite(data)
    data = data[index]
    if len(data) < 3:
        return [None], [None]
# Calculate Fourier transform
    FT = np.fft.fft(data)
    FT *= 2 / len(data)
    FT[0] /= 2.0
    Fabs = abs(FT)
    k = np.fft.fftfreq(len(data), d=d) * 2.0 * np.pi
# Plot data only up to coefficient (n_coef)
    eps = 1e-10
# Return log of power spectrum (add epsilon to avoid log(0)), wave numbers
    return np.log(Fabs[1:n_coef] ** 2 + eps), k[1:n_coef]


def spector_line(data, d, n_coef, half_width, iline):
    """
    Calculate depth of random sources with formula of (Spector and Grant,
    Geophysics, 1970) for one line.
    Depths are calculated by fitting two lines to logarithmic spectrum. The
    break point between the two lines is searched between the 4th spectral
    coefficient and the one at position n_Ny-4.

    Parameters
    ----------
    data : numpy 1D array, float
        Data to be analyzed
    d : float
        distance between data points [m].
    n_coef : int
        Number of coefficients to be used for line fitting.
    half_width : int
        Used for determination of local maxima: if the value of point # i
        is larger than all values between i-helf_width a,d i+half_width,
        the point i is considered a local maximum.

    Returns
    -------
    float
        Depth calculated with slope of small wave numbers.
    float
        Depth calculated with slope of large wave numbers.
    int
        Number of spectral coefficient where the slope break is located.
    float
        Intercept of first line (small wave numbers).
    float
        Intercept of second line (large wave numbers).
    float
        Misfit of data adjustment
    numpy 1D array, float
        Logarithmic power spectral values.
    numpy 1D array, float
        Wave numbers of power spectrum.

    """
    dd, kk = log_spect(data, d, n_coef)
    if not dd[0]:
        return None, None, -1, None, None, None, [None], [None], [None], [None]
    if kk[-1] < 0:
        index = kk > 0
        dd = dd[index]
        kk = kk[index]
# In order to avoid negative depths, the analysis is started at the coefficient
#    having the maximum amplitude (excluding the first coefficient which is the
#    average value)
    max_pos, d, _, _ = utils.min_max(dd, half_width=half_width)
    if len(max_pos) < 8 or np.max(max_pos) < int(len(dd/2.)):
        kkk = kk
        d = dd
    else:
        kkk = kk[max_pos]
    n0 = np.argmax(d)
    n1 = n0 + 3
    n2 = len(d) - 3
    if n2 <= n1:
        return None, None, -1, None, None, None, [None], [None], [None], [None]
# Fit two regression lines to data. For this, search breaking point
#     between third and 11th data point for which the fit is best
    reg1, reg2, isp, fit, slopes1, slopes2, inter1, inter2, fits, isplits = (
        utils.fit2lines(kkk, d, n0, n1, n2, True))
    # if iline == 18:
    #     with open("test.dat", "w") as fo:
    #         fo.write(f"{isp}  {reg1.coef_[0]}  {reg1.intercept_}  "
    #                  + f"{reg2.coef_[0]}  {reg2.intercept_}  {fit}\n")
    #         for i in range(len(fits)):
    #             fo.write(f"{isplits[i]}  {slopes1[i]}  {inter1[i]}"
    #                      + f"  {slopes2[i]}  {inter2[i]}   {fits[i]}\n")
    isplit = np.argmin(abs(kk - kkk[isp]))
    depth1 = -reg1.coef_[0] / 2.0
    depth2 = -reg2.coef_[0] / 2.0
    return depth1, depth2, isplit, reg1.intercept_, reg2.intercept_, fit, dd, \
        kk, d, kkk


def spector1D(data, direction, half_width):
    """
    Calculate depth of random sources with formula of (Spector and Grant,
    Geophysics, 1970) for all lines (N-S or E-W direction).
    Depths are calculated by fitting two lines to logarithmic spectrum. The
    break point between the two lines is searched between the 4th and the
    10th spectral coefficient.

    Parameters
    ----------
    data : object of class data
        Data to be treated
    direction : int
        if 0, analyze lines in Y direction, else in X-direction
    half_width : int
        A local maximum is detected at point i if
        value[i] = max(value[i-half_width:i+half_width])

    Returns
    -------
    lpos : numpy 1D float array of length number_of-lines
        Coordinate of the line (x_coordinate for N-S lines and vice versa)
    depths1 : numpy 1D float array of length number_of-lines
        Calculated average largest depth for each line (low frequencies)
    depths2 : numpy 1D float array of length number_of-lines
        Calculated average smallest depth for each line (high frequencies)
    intercepts1 : numpy 1D float array of length number_of-lines
        Intercepts for spectral fit of low frequencies
    intercepts2 : numpy 1D float array of length number_of-lines
        Intercepts for spectral fit of high frequencies
    isplits : numpy 1D int array of length number_of-lines
        Number of Fourier coefficients where the slope break is located
    fit : numpy 1D float array of length number_of-lines
        Misfit of overall adjustment for each line
    n_Ny : int
        Nyquist number for spectral calculations
    dsamp : sampling step along lines

    """
    ndat = np.array([len(data.y_inter), len(data.x_inter)])
    Ny = np.array(ndat / 2, dtype=int)
# Set parameters depending on measurement direction
    dx = data.x_inter[1] - data.x_inter[0]
    dy = data.y_inter[1] - data.y_inter[0]
    if direction == 0:
        nlines = len(data.x_inter)
        dsamp = dy
    else:
        nlines = len(data.y_inter)
        dsamp = dx
    n_Ny = Ny[direction]
# Prepare lists where to store the calculation results
    depths1 = []
    depths2 = []
    intercepts1 = []
    intercepts2 = []
    isplits = []
    lpos = []
    fits = []
# Loop over all lines, extract data and define coordinates
    for il in range(nlines):
        if direction:
            line_data = data.sensor1_inter[il, :]
            pos_line = data.y_inter[il]
        else:
            line_data = data.sensor1_inter[:, il]
            pos_line = data.x_inter[il]
        if il == 18:
            pass
        depth1, depth2, isplit, intercept1, intercept2, fit, _, _, _, _ =\
            spector_line(line_data, dsamp, n_Ny, half_width, il)
    # Store best fitting depths in list
        if not depth1:
            depths1.append(np.nan)
            depths2.append(np.nan)
            intercepts1.append(np.nan)
            intercepts2.append(np.nan)
            isplits.append(0)
            lpos.append(np.nan)
            fits.append(np.nan)
        else:
            depths1.append(max(depth1, depth2))
            depths2.append(min(depth1, depth2))
            intercepts1.append(intercept1)
            intercepts2.append(intercept2)
            isplits.append(isplit)
            lpos.append(pos_line)
            fits.append(fit)
    return (np.array(lpos), np.array(depths1), np.array(depths2),
            np.array(intercepts1), np.array(intercepts2), np.array(isplits),
            np.array(fits), n_Ny, dsamp)


def spector2D(data_c):
    """
    Calculate depth of random sources with formula of (Spector and Grant,
    Geophysics, 1970) in 2D.

    Choose first a window length. Spectral coefficients are averaged over
    radial equidistant coefficients. Depths are calculated by fitting two
    lines to logarithmic spectrum. The break point between the two lines is
    searched between the 4th and the 10th spectral coefficient.
    Results of all lines are saved in file spector.dat.

    The window length should be defined such that the number of Fourier
    coefficients is at least 8:
    (n = window_length/(2*max(dx,dy)), dx, dy, step sizes defined during
    interpolation)

    Results are stored in file spector2D_<data_type>.dat; Data type may be
    "magnetic" or "gravity".

    """
    x_min = data_c.x_inter.min()
    x_max = data_c.x_inter.max()
    dx = data_c.x_inter[1] - data_c.x_inter[0]
    y_min = data_c.y_inter.min()
    y_max = data_c.y_inter.max()
    dy = data_c.y_inter[1] - data_c.y_inter[0]
    d = max(dx, dy)
    max_len_x = x_max - x_min
    max_len_y = y_max - y_min
    window_len = np.round(min(max_len_x, max_len_y) / 4, 0)
    xstart = x_min + window_len / 2.0
    xend = x_max - window_len / 2.0
    nx = int((xend - xstart) / d)
    ystart = y_min + window_len / 2.0
    yend = y_max - window_len / 2.0
    ny = int((yend - ystart) / d)
    ntot = nx * ny
    nfac = max(int(np.ceil(np.sqrt(ntot / 500))), 1)
    # Get parameters
    step = d * nfac
    n_Nys = [int(window_len / (2 * dy)), int(window_len / (2 * dx))]
    ret, window_len, step, half_width, n_Ny = comm.get_spector2D(
        window_len, step, n_Nys)
    if not ret:
        print("No FFT analysis done")
        return (False, None, None, None, None, None, None, None, None, None,
                None, None, None, None)
    n_Nys = [int(window_len / (2 * dy)), int(window_len / (2 * dx))]
# Check whether given parameters are compatible
    if n_Ny < 8:
        _ = QtWidgets.QMessageBox.warning(
            None, "Warning",
            "For automatic depth determination the number of\n"
            + "FFT coefficients must be >= 8\nActual value: "
            + f"N_coef: {n_Ny}\n\nSpector2D not calculated\n",
            QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
        return (False, None, None, None, None, None, None, None, None, None,
                None, None, None, None)
    if n_Ny > n_Nys[0] or n_Ny > n_Nys[1]:
        _ = QtWidgets.QMessageBox.warning(
            None, "Warning",
            "For automatic depth determination the number of\n"
            + "FFT coefficients used for depth determination\n"
            + "must be <= nr of Nyquist coefficient\nActual values:\n"
            + f"Needed number of coefficients: {n_Ny}\n"
            + f"Nyquist in Y direction: {n_Nys[0]}\n"
            + f"Nyquist in X direction: {n_Nys[1]}\n\n"
            + "Retry increasing window length or interpolate with smaller"
            + " dy or dx\n\nSpector2D not calculated\n",
            QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
        return (False, None, None, None, None, None, None, None, None, None,
                None, None, None, None)
# Prepare lists where to store the calculation results
    nr, nc = data_c.sensor1_inter.shape
    nstep_x = int(step / dx)
    nwin_x = int(window_len / dx)
    if nwin_x % 2:
        nwin_x -= 1
    window_len_x = nwin_x * dx
    nwinx2 = int(window_len_x / (2 * dx))
    xstart = x_min + window_len_x / 2.0
    n_xstart = int((xstart - data_c.x_inter[0]) / dx)
    xend = x_max - window_len_x / 2.0
    xcalc_pos = np.arange(xstart, xend + step / 2.0, step)
    nstep_y = int(step / dy)
    nwin_y = int(window_len / dy)
    if nwin_y % 2:
        nwin_y -= 1
    window_len_y = nwin_y * dy
    nwiny2 = int(window_len_y / (2 * dy))
    ystart = y_min + window_len_y / 2.0
    n_ystart = int((ystart - data_c.y_inter[0]) / dy)
    yend = y_max - window_len_y / 2.0
    ycalc_pos = np.arange(ystart, yend + step / 2.0, step)
    nx_calc = len(xcalc_pos)
    ny_calc = len(ycalc_pos)
    depths_1 = np.zeros((ny_calc, nx_calc))
    depths_2 = np.zeros((ny_calc, nx_calc))
    intercepts_1 = np.zeros((ny_calc, nx_calc))
    intercepts_2 = np.zeros((ny_calc, nx_calc))
    depths1 = np.zeros((ny_calc, nx_calc))
    depths2 = np.zeros((ny_calc, nx_calc))
    depths3 = np.zeros((ny_calc, nx_calc))
    depths4 = np.zeros((ny_calc, nx_calc))
    intercepts1 = np.zeros((ny_calc, nx_calc))
    intercepts2 = np.zeros((ny_calc, nx_calc))
    intercepts3 = np.zeros((ny_calc, nx_calc))
    intercepts4 = np.zeros((ny_calc, nx_calc))
    isplits1 = np.zeros((ny_calc, nx_calc), dtype=int)
    isplits2 = np.zeros((ny_calc, nx_calc), dtype=int)
    fits = np.zeros((ny_calc, nx_calc))
    xpos = np.zeros(nx_calc)
    ypos = np.zeros(ny_calc)
    ii = -1
    i = n_xstart - nstep_x
    # for i in range(n_xstart, n_xend+1, nstep_x):
    for xx in xcalc_pos:
        i += nstep_x
        n1x = i - nwinx2
        n2x = i + nwinx2
        ii += 1
        jj = -1
        j = n_ystart - nstep_y
        xpos[ii] = data_c.x_inter[i]
        # for j in range(n_ystart, n_yend+1, nstep_y):
        for yy in ycalc_pos:
            j += nstep_y
            jj += 1
            n1y = j - nwiny2
            n2y = j + nwiny2
            ypos[jj] = data_c.y_inter[j]
            data = data_c.sensor1_inter[n1y:n2y, i]
            depth1, depth2, isplit1, intercept1, intercept2, fit1, dd, kk, _, \
                _ = (spector_line(data, dy, n_Ny, half_width, j))
            depths1[jj, ii] = depth1
            depths2[jj, ii] = depth2
            intercepts1[jj, ii] = intercept1
            intercepts2[jj, ii] = intercept2
            isplits1[jj, ii] = isplit1
            data = data_c.sensor1_inter[j, n1x:n2x]
            depth3, depth4, isplit2, intercept3, intercept4, fit2, dd, kk, _, \
                _ = (spector_line(data, dx, n_Ny, half_width, i))
            depths3[jj, ii] = depth3
            depths4[jj, ii] = depth4
            intercepts3[jj, ii] = intercept3
            intercepts4[jj, ii] = intercept4
            isplits2[jj, ii] = isplit2
            if depth1:
                if depth3:
                    depths_1[jj, ii] = (depth1 + depth3) * 0.5
                    depths_2[jj, ii] = (depth2 + depth4) * 0.5
                    intercepts_1[jj, ii] = (intercept1 + intercept3) * 0.5
                    intercepts_2[jj, ii] = (intercept2 + intercept4) * 0.5
                    fits[jj, ii] = np.sqrt((fit1**2 + fit2**2) * 0.5)
                else:
                    depths_1[jj, ii] = depth1
                    depths_2[jj, ii] = depth2
                    intercepts_1[jj, ii] = intercept1
                    intercepts_2[jj, ii] = intercept2
                    fits[jj, ii] = fit1
            elif depth3:
                depths_1[jj, ii] = depth3
                depths_2[jj, ii] = depth4
                intercepts_1[jj, ii] = intercept3
                intercepts_2[jj, ii] = intercept4
                fits[jj, ii] = fit2
    return (True, xpos, ypos, depths_1, depths_2, intercepts_1, intercepts_2,
            fits, window_len, nwiny2, nwinx2, step, half_width, n_Ny)


def gradient(data, dx, dy, filt=5.0):
    """
    Calculate absolute gradient of a data set interpolated onto a regular
    grid.

    The grid step may be different in x and y directions.

    Parameters
    ----------
    data : 2D numpy float array
        Data for which gradient should be calculated.
    dx : float
        Grid step in x-direction.
    dy : float
        Grid step in y-direction.
    filt : float
        Size of gaussian filter applied to data before gradient calculation
        in number of grid points (the maximum grid size from x and y
        direction is the reference). If filt==0, no gaussian filter applied

    Returns
    -------
    2D numpy float array with the same size as data
        Absolute data gradient.

    """
    import scipy.ndimage as nd

# Apply gaussian filter
    if dx > dy:
        sigx = filt
        sigy = sigx * dx / dy
    else:
        sigy = filt
        sigx = sigy * dy / dx
    sigma = [sigy, sigx]
    if sigx > 0.0:
        ny, nx = data.shape
        d = np.zeros((ny + 10, nx + 10))
        d[5:-5, 5:-5] = data
        for i in range(5):
            d[i, 5:-5] = data[0, :]
            d[ny - 1 - i, 5:-5] = data[-1, :]
        for i in range(5):
            d[:, i] = d[:, 5]
            d[:, nx + 9 - i] = d[:, -6]
        d = nd.filters.gaussian_filter(d, sigma, mode="constant")
        d = d[5:-5, 5:-5]
    else:
        d = np.copy(data)
    gx = np.zeros_like(d)
    gy = np.zeros_like(d)
    gx[:, 1:-1] = (d[:, 2:] + d[:, 0:-2] - 2 * d[:, 1:-1]) / (2 * dx)
    gx[:, 0] = gx[:, 1]
    gx[:, -1] = gx[:, -2]
    gy[1:-1, :] = (d[2:, :] + d[0:-2, :] - 2 * d[1:-1, :]) / (2 * dy)
    gy[0, :] = gy[1, :]
    gy[-1, :] = gy[-2, :]
    return np.sqrt(gx**2 + gy**2)
