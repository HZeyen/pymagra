# -*- coding: utf-8 -*-
"""
Last modified: Apr 23, 2025

@author: Hermann Zeyen
University Paris-Saclay

    functions:
       - magnetization_components
       - tandet
       - matrixExtension
       - gradient
       - integral
       - compon
       - mag_color_map
       - data_plot

    Class: Earth_mag with methods:
          - __init__
          - earth_components

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def magnetization_components(sus, rem, inc, dec, earth):
    """
    Routine calculates the 3 components (tx, ty, tz) of magnetization for
    a combination of induced and remanent magnetization.

    """
    tx = 0.0
    ty = 0.0
    tz = 0.0
    if sus != 0.0 and earth.f > 0.0:
        mag = earth.f * sus
        tx = mag * earth.cdci
        ty = mag * earth.sdci
        tz = mag * earth.sie
# The factor 100 comes from the fact that the calculated anomaly in A/m must be
# multiplied by mu0/4pi = 10**-7 and again by 10**9 to pass from T to nT.
    if rem != 0:
        rem_gauss = rem * 100.0
        cdr = np.cos(np.radians(dec))
        sdr = np.sin(np.radians(dec))
        cir = np.cos(np.radians(inc))
        sir = np.sin(np.radians(inc))
        ehr = rem_gauss * cir
        tx += ehr * cdr
        ty += ehr * sdr
        tz += rem_gauss * sir
    return tx, ty, tz


def tandet(a, b, t_sum):
    """
    arctan argument addition

    Input:
        a,b : float
                atan arguments to be added
                negative a for subtraction
        t_sum: float
                multiple of pi/2 to be added coming eventually from
                an earlier call

    Returns:
        s: float
            value of summed argument
        t_sum: float
            multiple of pi/2 to be added

    """
    q1 = a + b
    q2 = 1.0 - a * b
    if np.isclose(q2, 0.0):
        if q1 < 0.0:
            t_sum -= np.pi / 2.0
        else:
            t_sum += np.pi / 2.0
        s = 0.0
    else:
        s = q1 / q2
        if q2 < 0.0:
            if q1 < 0.0:
                t_sum -= np.pi
            elif q1 > 0.0:
                t_sum += np.pi
    return s, t_sum


def matrixExtension(data):
    """
    Creation of extended matix for 2D Fourier transform.
    The toutine mirrors the lower half of the matrix and adds it at the bottom
    and mirrors the upper half to the top. Equivalently right and left

    Input:
        data : 2D numpy array with data

    Output:
        d : 2D numpy array extended in both directions
        (ny1,nx1): Tuple with starting indices of the original data in matrix d
        (ny2,nx2): Tuple with final indices of the original data in matrix d
        plus one. The original data may thus be retrieved as
        data = d[ny1:ny2,nx1:nx2]

    """
    ny, nx = data.shape
    nx_add = int((nx - 2) / 2)
    ix_add_right = nx - nx_add - 1

    ny_add = int((ny - 2) / 2)
    d = np.zeros((ny + 2 * ny_add, nx + 2 * nx_add))
    iy_add_up = d.shape[0] - ny_add - 1

    nx_right = nx_add + nx
    ny_up = ny_add + ny
    edge = np.mean(
        np.array(list(data[:, 0]) + list(data[0, :]) + list(data[-1, :])
                 + list(data[:, -1])))
    d[ny_add:ny_up, nx_add:nx_right] = data
    fac = (np.sin(np.arange(nx_add) * np.pi / (2 * nx_add))) ** 2
    fac2 = np.flip(fac)
    j = -1
    for i in range(ny_add, ny_up):
        j += 1
        d[i, :nx_add] = (fac*(data[j, 1]-edge)*np.flip(data[j, 1:nx_add+1])
                         + edge)
        d[i, nx_right:] = (fac2*(data[j, -2]-edge)
                           * np.flip(data[j, ix_add_right:nx-1])+edge)
    fac = (np.sin(np.arange(ny_add) * np.pi / (2 * ny_add))) ** 2
    fac2 = np.flip(fac)
    for i in range(d.shape[1]):
        d[:ny_add, i] = fac*(d[1, i]-edge)*np.flip(d[1:ny_add+1, i])+edge
        d[ny_up:, i] = fac2*(d[-2, i]-edge)*np.flip(d[iy_add_up:-1, i])+edge
    return d, (ny_add, nx_add), (ny_add + ny, nx_add + nx)


def gradient(data, dx, dy, direction):
    """
    Routine calculates the derivative of a 2D array using FFT
    For x derivative, coefficients are multiplied with 1j*kx,
    for y derivatives with 1j*ky and for z derivative with np.sqrt(kx**2+ky**2)

    Parameters
    ----------
    data : numpy 2D float array (nr*nc)
        Data to be derivated
    dx : float
        grid step size in x direction.
    dy : float
        grid step size in y direction.
    direction : str, may be "x", "y" or "z"
        Direction in which the dervivative is calculated

    Returns
    -------
    numpy 2D float array with derivatives
        DESCRIPTION.

    """
    d, corner1, corner2 = matrixExtension(data)
    dF = np.fft.fft2(d)
    ny1 = corner1[0]
    nx1 = corner1[1]
    ny2 = corner2[0]
    nx2 = corner2[1]
    ny, nx = d.shape
    kx = np.fft.fftfreq(nx, dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, dy) * 2.0 * np.pi
    if direction.lower() == "x":
        s = np.outer(np.ones_like(ky), kx) * 1j
    elif direction.lower() == "y":
        s = np.outer(ky, np.ones_like(kx)) * 1j
    else:
        u = np.outer(np.ones_like(ky), kx)
        v = np.outer(ky, np.ones_like(kx))
        s = np.sqrt(u**2 + v**2)
    dF *= s
    return np.real(np.fft.ifft2(dF)[ny1:ny2, nx1:nx2])


def integral(data, dx, dy, direction):
    """
    Routine calculates the integral of a 2D array using FFT
    For x derivative, coefficients are divided by 1j*kx,
    for y derivatives wby 1j*ky and for z derivative by np.sqrt(kx**2+ky**2)
    Zero dividers (zero wavenumbers) are set to 1.

    Parameters
    ----------
    data : numpy 2D float array (nr*nc)
        Data to be derivated
    dx : float
        grid step size in x direction.
    dy : float
        grid step size in y direction.
    direction : str, may be "x", "y" or "z"
        Direction in which the dervivative is calculated

    Returns
    -------
    numpy 2D float array with derivatives
        DESCRIPTION.

    """
    d, corner1, corner2 = matrixExtension(data)
    dF = np.fft.fft2(d)
    ny1 = corner1[0]
    nx1 = corner1[1]
    ny2 = corner2[0]
    nx2 = corner2[1]
    ny, nx = d.shape
    kx = np.fft.fftfreq(nx, dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, dy) * 2.0 * np.pi
    if direction.lower() == "x":
        s = np.outer(np.ones_like(ky), kx) * 1j
    elif direction.lower() == "y":
        s = np.outer(ky, np.ones_like(kx)) * 1j
    else:
        u = np.outer(np.ones_like(ky), kx)
        v = np.outer(ky, np.ones_like(kx))
        s = np.sqrt(u**2 + v**2)
    s[s == 0] = 1
    dF /= s
    return np.real(np.fft.ifft2(dF)[ny1:ny2, nx1:nx2])


def compon(dx, dy, dz, earth):
    """
    Calculate the full horizontal and vertical components of rock field from
    the three directional components

    Parameters
    ----------
    dx : float
        field anomaly in x (NS) direction
    dy : float
        field anomaly in y (EW) direction
    dz : float
        field anomaly in z direction
    earth : class Earth_mag object

    Returns
    -------
    dh : float
        horizontal component of the anomaly
    dt : float
        total field component of the anomaly

    """
    # If effect of body is large with respect toe Earth's field, the next three
    # lines should be uncommented and the other two lines commented.
    dh = dx * earth.cde + dy * earth.sde
    dt = dx * earth.cdci + dy * earth.sdci + dz * earth.sie
    return dh, dt


def mag_color_map(vmn, vmx, cif=2):
    """
    Set colormap for magnetic data such that white corresponds to 0, but the
    blue and red parts of the colorbar are plotted proportional to the values
    and not, as by default, proportional to the colomap

    Parameters
    ----------
    vmn : float
        Minimum value of data for color scale
    vmx : flag
        Maximum value of data for color scale
    cif : int, optional
        number of ciphers to round to. The default is 2.

    Returns
    -------
    br_map : Matplotlib color scale
        Color scale to be used in magnetic anomaly map plotting
    norm : matplotlib norm corresponding to color scale

    """
    c = cif
# Find number of ciffers equal to or above cif necessary to distinguish
# minimum from maximum rounded value
    while True:
        fac = 10**c
        av = np.round((vmn + vmx) / 2.0, c)
        vmin = np.ceil((vmn - av) * fac) / fac
        vmax = np.floor((vmx - av) * fac) / fac
        if vmax > vmin:
            break
        c += 1
    vmin += av
    vmax += av
    if np.isclose(vmax, vmin):
        vmin -= 0.5
        vmax += 0.5
    ncol_tot = 1024
    if vmax <= 0.0:
        all_colors = plt.cm.Blues_r(np.linspace(0, 1, ncol_tot))
    elif vmin >= 0.0:
        all_colors = plt.cm.Reds(np.linspace(0, 1, ncol_tot))
    else:
        ncol_neg = int(-ncol_tot * vmin / (vmax - vmin))
        ncol_pos = ncol_tot - ncol_neg
        # Define blue colormap for negative values
        colors_neg = plt.cm.Blues_r(np.linspace(0, 1, ncol_neg))
        # Define red colormap for positive values
        colors_pos = plt.cm.Reds(np.linspace(0.05, 1, ncol_pos))
        all_colors = np.vstack((colors_neg, colors_pos))
    br_map = colors.LinearSegmentedColormap.from_list("blue_red", all_colors)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)
    return br_map, norm


def data_plot(data, fig, ax, title="", xtitle="Easting [m]",
              ytitle="Northing [m]", cmap="rainbow", norm=None, extent=None,
              cbar_title="", vmin=None, vmax=None, font_ax=12, font_tit=14):
    """
    Plots a raster image of regularly gridded data

    Parameters
    ----------
    data : numpy 2D array of size [ny,nx]
        data to be plotted it is supposed that data[0,0] is in the lower left
        corner (therefore the use of np.flip)
    ax : matplotlib axis
        Axis where to plot the map
    title : str, optional
        Title of the plot. The default is "".
    xtitle : str, optional
        Label of the horizontal axis. The default is "Easting [m]".
    ytitle : str, optional
        Label of the vertical axis. The default is "Northing [m]".
    cmap : matplotlib color map, optional
        Colormap to use. The default is "rainbow".
    norm : matplotlib color norm, optional
        Norm to be used, if given. The default is None.
    extent : list, optional
        limits of the axis labels: [xmin,xmax,ymin,ymax]
        The default is None.
        If None, the limits are set to _0.5 for both directions and to nx-0.5
            for the horizontal axis, ny-0.5 for the vertical axis.
        In order to plot the block limits at the correct position, the lower
            limit ust be 1/2 block width smaller than the minimum coordinate of
            the block centers and the maximum 1/2 block width larger than the
            maximum coordinate of block centers.
    cbar_title : str, optional
        Label of the color bar . The default is "".
    vmin : float, optional
        Minimum value of color bar. The default is None.
        If None, minimum value of data is used
    vmax : float, optional
        Maximum value of color bar. The default is None.
        If None, maximum value of data is used
    font_ax : int, optional. Default: 12
        Font size for axis and color bar annotation
    font_tit : int, optional. Default: 14
        Font size for figure title

    Returns
    -------
    im : matplotlab image
    cbar : matplotlib color bar

    """
    if not extent:
        shape_y, shape_x = data.shape
        extent = [-0.5, shape_x - 0.5, 0.5, shape_y - 0.5]
    if not vmin:
        vmin = np.round(data.min(), 1)
    if not vmax:
        vmax = np.round(data.max(), 1)
    if norm:
        im = ax.imshow(np.flip(data, axis=0), norm=norm, cmap=cmap,
                       aspect="equal", extent=extent, rasterized=True)
    else:
        im = ax.imshow(np.flip(data, axis=0), cmap=cmap, aspect="equal",
                       extent=extent, rasterized=True, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=font_tit)
    ax.set_xlabel(xtitle, fontsize=font_ax)
    ax.set_ylabel(ytitle, fontsize=font_ax)
    ax.tick_params(labelsize=font_ax)
    # cax = ax.inset_axes([0.95, 0.1, 0.05, 0.9])
    cax = ax.inset_axes([1.05, 0.05, 0.05, 0.9], transform=ax.transAxes)
    cbar = fig.colorbar(im, shrink=0.9, cax=cax)
    cbar.set_label(cbar_title, fontsize=font_ax - 2)
    cbar.ax.tick_params(labelsize=font_ax - 2)
    cbar.ax.set_ylabel(cbar_title, size=10)
    cbar.ax.text(0.0, -0.02, f"{vmin}", verticalalignment="top",
                 horizontalalignment="left", transform=cbar.ax.transAxes,
                 fontsize=10)
    cbar.ax.text(0.0, 1.02, f"{vmax}", verticalalignment="bottom",
                 horizontalalignment="left", transform=cbar.ax.transAxes,
                 fontsize=10)
    return im, cbar


def get_extremes(data, width=5):
    """
    Search position of relative minima and maxima on a 1D or2D grid

    Parameters
    ----------
    data : 1D or 2D numpy float array
        Data to be analyzed
    width : int, optional, default: 5
        A maximum is recognized if the value at a point (i,j) is larger than
        or equal to all other values within an area of
        (i-width:i+width, j-width:j+width)

    Returns
    -------
    min_pos : list of tuples
        tuples (i,j) of relative minima positions within array data
        (i is y-direction, j is x-direction.
    max_pos : list of tuples
        Similar to min_pos but for relative maxima.

    """
    max_pos = []
    min_pos = []
    if len(data.shape) == 1:
        nx = len(data)
        for ix in range(width, nx, 2 * width):
            d = data[ix - width: ix+width+1]
            pos = np.where(d == np.nanmax(d))
            xmax = ix + pos[0][0] - width
            dmax = data[xmax]
            x1 = max(xmax - width, 0)
            x2 = min(xmax + width, nx)
            dd = data[x1:x2]
            if dd.max() == dmax:
                max_pos.append((xmax,))
            pos = np.where(d == np.nanmin(d))
            xmin = ix + pos[0][0] - width
            x1 = max(xmin - width, 0)
            x2 = min(xmin + width, nx)
            dmin = data[xmin]
            dd = data[x1:x2]
            if dd.min() == dmin:
                min_pos.append((xmin,))
        return min_pos, max_pos
    ny, nx = data.shape
    for ix in range(width, nx, 2 * width):
        for iy in range(width, ny, 2 * width):
            d = data[iy-width: iy+width+1, ix-width: ix+width+1]
            pos = np.where(d == np.nanmax(d))
            ymax = iy + pos[0][0] - width
            xmax = ix + pos[1][0] - width
            y1 = max(ymax - width, 0)
            y2 = min(ymax + width, ny)
            x1 = max(xmax - width, 0)
            x2 = min(xmax + width, nx)
            dmax = data[ymax, xmax]
            dd = data[y1:y2, x1:x2]
            if dd.max() == dmax:
                max_pos.append((ymax, xmax))

            pos = np.where(d == np.nanmin(d))
            ymin = iy + pos[0][0] - width
            xmin = ix + pos[1][0] - width
            y1 = max(ymin - width, 0)
            y2 = min(ymin + width, ny)
            x1 = max(xmin - width, 0)
            x2 = min(xmin + width, nx)
            dmin = data[ymin, xmin]
            dd = data[y1:y2, x1:x2]
            if dd.min() == dmin:
                min_pos.append((ymin, xmin))
    return min_pos, max_pos
