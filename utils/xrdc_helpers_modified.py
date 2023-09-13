"""
All code here is taken from Oliver Hoidn's package here: https://github.com/hoidn/xrd_clustering
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.fft import fft, ifft,fftshift, ifftshift, fftn, ifftn
from scipy.signal import blackman as blackman_window
from scipy.ndimage import gaussian_filter as gf
from scipy.interpolate import interp1d
from scipy.interpolate import NearestNDInterpolator
from scipy import ndimage as nd

def power(arr):
    """
    Return squared amplitude of an array.
    """
    ampsq = arr * np.conjugate(arr)
    return np.real(ampsq)

def filter_bg(pattern, smooth = 1.5, invert = False, **kwargs):
    """
    Extract high-frequency component from a 1d spectrum by high-pass
    filtering with a 1 - Blackman window, taking the IFFT squared
    amplitude, and applying gaussian smoothing.
    """
    blackman = blackman_window(len(pattern))
    # high-pass filtered diffraction pattern
    fastq_indicator = power(ifft(ifftshift((1 - blackman) * fftshift(fft(pattern)))))

    return gf(fastq_indicator, smooth)


def extract_single(row, q_cutoff = .001, smooth_q = 1.7):
    """
    Default procedure for extracting the high-frequency component of a
    single 1d diffraction pattern.
    """
    return filter_bg(row, smooth_q, q_cutoff = q_cutoff)

def apply_bottom(func, arr, axis = None, **kwargs):
    """
    apply 1d array-transforming function to bottom (q) dimension
    """
    def new_f(*args):
        """ bind kwargs """
        return func(*args, **kwargs)

    if axis is None:
        axis = len(arr.shape) - 1
    return np.apply_along_axis(new_f, axis, arr)

def mk_smooth(patterns, smooth_neighbor, smooth_q):
    n = len(patterns.shape)
    return (smooth_neighbor,) * (n - 1) + (smooth_q,)

def reference_bgsub(patterns, smooth_q = 1.7, smooth_neighbor_background = 1,
        q_cutoff = .001, **kwargs):
    """
    Extract high-frequency component (in q) from a 2d XRD dataset. This
    method distorts peak intensities but is good at identifying their
    locations.
    """
    bgsubbed_nosmooth = apply_bottom(extract_single, patterns,
        q_cutoff = q_cutoff, smooth_q = smooth_q)
    bgsubbed_final = gf(bgsubbed_nosmooth, mk_smooth(patterns, smooth_neighbor_background, smooth_q))
    bgsubbed_final *= patterns.max() / bgsubbed_final.max() #np.percentile(patterns, 99.9) / np.percentile(bgsubbed_final, 99.9)
    return bgsubbed_final

def interprows(arr, mask, fn = None, **kwargs):
    # TODO refactor, generalize
    if fn is None:
        def fn(*args):
            return interp1d(*args, bounds_error = False, **kwargs)
    if len(arr.shape) == 2:
        res = []
        for row, rowmask in zip(arr, mask):
            x = np.indices(row.shape)[0][rowmask]
            y = row[rowmask]
            if len(x) >= 2:
                f = fn(x, y)
                res.append(f(np.indices(row.shape)[0]))
            else:
                if len(y) > 0:
                    res.append(np.repeat(max(y), len(row)))
                else:
                    res.append(np.repeat(0., len(row)))
        return np.vstack(res)
    elif len(arr.shape) == 3:
        res = np.zeros_like(arr)
        n, m, _ = arr.shape
        res = np.zeros_like(arr)
        for i in range(n):
            for j in range(m):
                row = arr[i, j, :]
                rowmask = mask[i, j, :]
                x = np.indices(row.shape)[0][rowmask]
                y = row[rowmask]
                f = fn(x, y)
                res[i, j, :] = f(np.indices(row.shape)[0])
        return res

def get_bgmask(patterns, threshold, **kwargs):
    """
    Find peak regions and return a mask that identifies them.
    Peak pixels map to False and background pixels map to True.

    This function returns a boolean array
    """
    bgsubbed = reference_bgsub(patterns, **kwargs)
    percentiles = np.percentile(bgsubbed, threshold, axis = len(patterns.shape) - 1)
    mask = (bgsubbed >
            percentiles[..., None])
    bgsubbed[mask] = np.nan
    bkgmask = ~np.isnan(bgsubbed)
    return bkgmask

def get_background_nan(patterns, threshold = 50, smooth_q = 1.7,
        smooth_q_background = 10, smooth_neighbor_background = 1, q_cutoff = .001):
    """
    Mask peak entries in patterns to np.nan and optionally do some Gaussian smoothing.
    """
    # TODO smooth or not?
    smooth = mk_smooth(patterns, smooth_neighbor_background, smooth_q_background)
    bkgmask = get_bgmask(patterns, threshold, smooth_q = smooth_q, smooth_neighbor_background = smooth_neighbor_background, q_cutoff = q_cutoff)
    filled_bg = interprows(patterns, bkgmask)
    smooth_bg = gf(filled_bg, smooth)
    return smooth_bg

def get_background(patterns, threshold = 50, bg_fill_method = 'simple',
        smooth_q = 1.7, smooth_neighbor_background = 1, q_cutoff = .001,
        smooth_q_background = 10, smooth_before = True, smooth_after = True):
    smooth = mk_smooth(patterns, smooth_neighbor_background, smooth_q)
    """
    If smooth_before, smooth background values before interpolation.
    If smooth_after, smooth background estimate post-interpolation.

    Background smoothing is applied *before* interpolation but not
    after. The returned background array is not smoothed.
    """
    if bg_fill_method in ['none', 'simple', 'extrap_1d']:
        smooth_bg = get_background_nan(patterns, threshold = threshold,
            smooth_q_background = 0,
            smooth_neighbor_background = 0, q_cutoff = q_cutoff)
        if bg_fill_method == 'none':
            mask = get_bgmask(patterns, threshold, smooth_q_background = 0,
                smooth_neighbor_background = 0, q_cutoff = q_cutoff)
            filled_data = smooth_bg
        elif bg_fill_method == 'simple':
            # TODO am i getting the higher-dimensional nearest neighbor?
            mask = np.where(~np.isnan(smooth_bg))
            interp = NearestNDInterpolator(np.transpose(mask), smooth_bg[mask])
            filled_data = interp(*np.indices(smooth_bg.shape))
        elif bg_fill_method == 'extrap_1d':
            filled_data = fill_nd(smooth_bg) 
        else:
            raise ValueError
    elif bg_fill_method == 'cloughtocher':
        mask = get_bgmask(patterns, threshold)
        filled_data = CTinterpolation(mask * patterns)
    else:
        raise ValueError
    if smooth_before is False:
        raise NotImplementedError
    if smooth_after:
        filled_data = gf(filled_data, mk_smooth(filled_data, smooth_neighbor_background, smooth_q_background))
    return filled_data

def gaussNd(sigma):
    """
    Returns a function that's a gaussian over a cube of coordinates
    of any dimension.
    """
    N = 1 / (sigma * np.sqrt(2 * np.pi))
    def f(*args):
        n = args[0].shape[0]
        def g(*args2):
            args_sq = (np.array(args2) * np.array(args2)).sum(axis = 0)
            return np.exp(- args_sq / (2 * sigma**2))
        # TODO check the offset
        x0 = (n  -1) / 2
        return N * g(*(arg - x0 for arg in args))
    return f

def gauss_low_Nd(arr, cutoff):
    # TODO assert cubic shape
    n = len(arr)
    args = np.indices(arr.shape)
    sigma = cutoff * n
    return (gaussNd(sigma)(*args))

def lowpassNd(arr, cutoff, mode = 'gaussian'):
    """
    Low pass filter with a circular step or gaussian frequency mask
    """
    if mode == 'step':
        raise NotImplementedError # TODO update this
        mask = draw_circle(arr, int(cutoff * ((arr.shape[0] + arr.shape[1]) / 2)))
    elif mode == 'gaussian':
        mask = gauss_low_Nd(arr, cutoff)
        mask /= mask.max()
    else:
        raise ValueError
    arrfft = fftshift(fftn(arr))
    arr_filtered = ifftn(ifftshift(mask * arrfft))
    return arr_filtered

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells by the value of the
    nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def separate_signal(patterns, cutoff = .2, mode = 'gaussian',
        background_after_filter = True, q_cutoff = .001, **kwargs):
    """
    Decompose a dataset into high- and low-frequency components in the
    non-q dimensions. Any rows that sum to zero are neglected.

    If background_after_filter, the background is estimated without
    first removing high-frequency components.

    The most important keyword arguments are:
        -cutoff: frequency cutoff for noise extraction in non-q dimensions.
        -threshold: percentage of pixels to use in the background
            interpolation. A lower value excludes more points in and
            surrounding peak regions and therefore gives a more conservative
            estimate of the background.
        -background_after_filter: **IMPORTANT** this can be set to True
            in the case of truly continuous datasets, but otherwise it
            must be False. Else noise removal will corrupt both the
            background estimate and the background-subtracted signal
            (fast_q).
        -smooth_q_background: gaussian smoothing standard deviation for
            the interpolated background. Should be smaller than the
            scale of background variations.
        -smooth_q: gaussian smoothing standard deviation for peak
            extraction (should be of order peak FWHM / 2).
        -smooth_neighbor_background: gaussian smoothing standard
            deviation for non-q dimensions of the background estimate.
            Can be of order unity for connected datasets but should
            be set to 0 if the background is discontinuous accross
            neighboring patterns.

    Other arguments:
        -mode: == 'gaussian' or 'step'; kernel to use for the non-q
            frequency filtering.
        -q_cutoff: frequency cutoff for q peak filtering (deprecated;
            should be a value close to 0 since peak extraction uses a
            Blackman window by default, which is sufficient on its own)
        -bg_fill_method. fill method for background values outside of
            interpolation range; should equal one of the following:
                'simple': nearest neighbor matching across non-q dimensions
                'none': np.nan values outside the interpolation range
                'extrap_1d': 1d extrapolation using the nearest non-nan
                    value
                'cloughtocher': cubic 2d interpolation

    Returns tuple:
        (interpolated background (excluding high-frequency non-q component),
        signal (excluding high-frequency non-q component) - interpolated background,
        low-frequency non-q signal,
        high-frequency non-q signal))
    """
    for ii, jj in zip((patterns.shape[0],) + patterns.shape[:-1], patterns.shape[:-1]):
        # array must be square for fourier filtering to work in the non-q dimensions
        assert ii == jj 
    # TODO filter T before background or after? need to do a more careful comparison
    # calculate the low-frequency component in xy
    nq = patterns.shape[-1]
    low_xy = np.zeros_like(patterns)
    wafer_mask = (patterns.sum(axis = (len(patterns.shape) - 1)) != 0)
    for i in range(nq):
        low_xy[..., i] = np.real(lowpassNd(fill(patterns[..., i], patterns[..., i] == 0), cutoff, mode)) * wafer_mask
    high_xy = patterns - low_xy

#    # TODO take cutoff parameter for q filtering as well
    if background_after_filter:
        interpolated_background = get_background(low_xy, q_cutoff = q_cutoff,
            **kwargs)
        fast_q = low_xy - interpolated_background
    else:
        interpolated_background = get_background(patterns, q_cutoff = q_cutoff,
            **kwargs)
        fast_q = patterns - interpolated_background
    return interpolated_background, fast_q, low_xy, high_xy

def separate_background(input_patterns,
                       plot=False,savePlot = False, title = None,savePath = os.getcwd(),
                       threshold = 15, smooth_q = 1.7, smooth_neighbor_background = 0,
                       smooth_q_background = 0,bg_smooth_post=20,
                       q = None):

    """
    This is a wrapper function for separate_signal() from the xrdc.sep submodule.
    It has some standard parameters for background estimation that are a good starting
    place for the pharmaceutical cocrystal data obtained at SSRL.

    :parameter input_patterns: a list of X-ray diffraction histograms. can also be a list of lists of XRD input_patterns
    :type input_parameters: list
    :parameters plot: option to plot the separated backgrounds on output
    :type plot: boolean
    :parameter savePlot: option to save the plots generated
    :type savePlot: boolean
    :parameter title: a list of titles to add to the plots
    :type title: None or list of strings
    :parameter savePath: path to save the generated plot
    :type savePath: path (str); defaults to the current directory
    :parameter threshold:
    :type threshold: float
    :parameter smooth_q:
    :type smooth_q: float
    :parameter smooth_neighbor_background:
    :type smooth_neighbor_background: float
    :parameter smooth_q_background:
    :type smooth_q_background: float
    :parameter bg_smooth_post:
    :type bg_smooth_post: float
    :parameter q: optional list of arrays to use as q-values for the plot (or two theta values, or any value you like)
    :type q: list of arrays, or list of list of arrays
    :return background: the estimated backgrounds for each input histogram
    :type background: list of arrays
    :return fast_q: the estimated crystalline peaks from each input histogram
    :type fast_q: list of arrays
    :return au: the estimated uncertainty in the background estimation
    :type au: array
    """

    # check if input_patterns is a list or not
    # if not, make it a list

    # if type(input_patterns) is not list:
    #     print('Not a list!')
    #     input_patterns = [input_patterns]

    background = []
    fast_q = []
    au = []

    # now do the background and noise estimation
    # for index,patterns in enumerate(input_patterns):
    #     print(index,patterns)

    # # sort the arrays based on their maximal intensity
    # maxes = [np.max(p) for p in patterns]

    # # get the indices for sorting so that you can undo it later
    # pinds = np.argsort(maxes)
    # reverse_pinds = np.argsort(pinds)
    # patterns = patterns[pinds]

    patterns = input_patterns
    tbackground, tfast_q, tslow_T, tfast_T = separate_signal(patterns,
                                    background_after_filter = False,
                                    threshold = threshold, smooth_q = smooth_q,
                                smooth_neighbor_background = smooth_neighbor_background,
                                smooth_q_background = smooth_q_background,
                                    bg_fill_method = 'simple')

    # estimate the uncertainty in the background estimation
    #aggregate_uncertainty = np.sqrt((fast_T.std(axis = 0)**2 + background.std(axis = 0)**2) / N)
    N = len(patterns)
    tau = np.sqrt((tfast_T.std(axis = 0)**2 + tbackground.std(axis = 0)**2) / N)
    tbackground = gf(tbackground, (0, bg_smooth_post))
    tau = gf(tau, bg_smooth_post)

    # get rid of any negative values
    tfast_q[tfast_q < 0] = 0

    # tfast_q = tfast_q[reverse_pinds]
    # tbackground = tbackground[reverse_pinds]

    background.append(tbackground)
    fast_q.append(tfast_q)
    au.append(tau)



    if plot == True:

        # check to see if they provided a q array
        # if not, create one
        if q == None:
            x = np.arange(len(patterns.mean(axis=0)))
        else:
            x = q

        # okay visualize the background estimation
        fig,ax = plt.subplots(1,2,figsize=(12,5))

        # make sure the pattern is centered around zero
        # subtract off the minimum intensity value to bring it down to zero
        patt = patterns.mean(axis=0) - np.min(patterns.mean(axis=0))

        # do the same with the background
        tbpatt = tbackground.mean(axis=0) - np.min(tbackground.mean(axis=0))

        ax[0].plot(x,patt,'k-',label='Raw Histogram')
        ax[0].plot(x,tbpatt,c='tab:orange',label=r'$I_{amorph}(q)$')
        #ax[0].plot(x,tau,c='tab:red',label='Estimated Uncertainty')
        #ax[0].plot(x,100*tau,'r-',label='Estimated Uncertainty X 100')
        ax[0].legend()

        ax[1].plot(x,tbpatt,c='tab:orange',label= r'$I_{amorph}(q)$')
        ax[1].plot(x,tfast_q.mean(axis=0),'b-',label= r'$I_{crys}(q)$')
        ax[1].legend()

        ax[0].set_xlabel(r'q ($\AA^{-1}$)')
        ax[1].set_xlabel(r'q ($\AA^{-1}$)')
        ax[0].set_ylabel('Intensity')
        ax[0].set_ylabel('Intensity')


        if title == None:
            pass
        else:
            plt.title(title[index])

        if savePlot == True:

            plt.savefig(savePath + title[index] + 'background_estimation.png')

        plt.show()

    return background,fast_q,au