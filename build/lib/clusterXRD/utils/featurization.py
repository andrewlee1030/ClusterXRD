import numpy as np
from scipy.integrate import simpson

final_features_and_weights = {
                            'I_max': 1, 
                            'I_mu': 1, 
                            'fwhm_max': 1, 
                            'fwhm_mu': 1, 
                            'bw_max': 1, 
                            'bw_mu': 1, 
                            'n_floating_peaks': 1,
                            'pse_max': 1, 
                            'pse_mu': 1,
                            'n_peaks_size_0': 1, 
                            'n_peaks_size_1': 1, 
                            'n_peaks_size_2': 1, 
                            'n_peaks_size_3': 1,
                            'n_peaks': 1,
                            'pct_area_under_raw_data': 1} # features actually used for clustering, along with their relative weights

feature_names = list(final_features_and_weights.keys())

def get_percentage_area_under_raw_data(histogram_intensities):
    '''
    Calculates the percentage of the histogram that is under the intensity curve

    Args:
        histogram_intensities (Numpy array): array of histogram intensities

    Returns:
        float: value between 0 and 1 representing the proportion under the intensity curve
    '''
    max_intensity = np.max(histogram_intensities)
    total_area = max_intensity * (len(histogram_intensities)-1)
    area_under_histogram_intensities = simpson(histogram_intensities)

    return area_under_histogram_intensities / total_area

def get_n_floating_peaks(crystalline_data,peak_data,min_threshold = 50):
    '''
    Count the number of peaks that have peak starts and ends above a certain intensity.

    Args:
        crystalline_data (numpy array): background subtracted histogram intensities
        peak_data (pandas DataFrame): peak starts, locations, and ends
        min_threshold (float): minimum intensity for a peak start or ends to count for a floating peak

    Returns:
        int: the number of floating peaks

    '''
    peak_start_intensities = crystalline_data[peak_data['Peak Start Index']]
    peak_end_intensities = crystalline_data[peak_data['Peak End Index']]

    starts_above_threshold = peak_start_intensities > min_threshold
    ends_above_threshold = peak_end_intensities > min_threshold

    both_above_threshold = starts_above_threshold & ends_above_threshold
    return np.sum(both_above_threshold)

def get_peak_start_end_intensities(crystalline_data,peak_data):
    '''
    Calculate the mean between peak starts and ends for all peaks in a histogram.

    Args:
        crystalline_data (numpy array): array of histogram intensities 
        peak_data (pandas DataFrame): peak starts, locations, and ends
    
    Returns:
        numpy array: intensities of each averaged peak start and end
    '''
    peak_start_intensities = crystalline_data[peak_data['Peak Start Index']]
    peak_end_intensities = crystalline_data[peak_data['Peak End Index']]

    averaged_intensities = np.mean(np.vstack([peak_start_intensities,peak_end_intensities]),axis=0)
    
    return averaged_intensities

def get_peak_base_widths(peak_data,crystalline_data,q_width):
    '''
    Returns the normalized width of all peaks in a histogram.

    Args:
        peak_data (pandas DataFrame): peak starts, locations, and ends
        crystalline_data (numpy array): array of histogram intensities
        q_width (float): width of histograms in q-units

    Returns:
        numpy array: normalized widths of each peak
    '''
    widths = peak_data['Peak End Index'] - peak_data['Peak Start Index']
    return widths / len(crystalline_data) / q_width # need to normalize by the total width of the plot AND the q_width

def get_derived_features(data):
    '''
    Calculate derived features given an array of raw feature values.

    Args:
        data (numpy array): raw feature values

    Returns:
        list: [25th percentile, 75th percentile, mean, maximum values] of input values
    '''
    pctile_25 = np.quantile(data,0.25)
    pctile_75 = np.quantile(data,0.75)
    mean = np.mean(data)
    max = np.max(data)

    return [pctile_25, pctile_75, mean, max]

def get_n_peaks_per_bin_size(peak_intensities,binned_intensities):
    '''
    Counts the number of peaks within specified intensity intervals. For example, when binned_intensities = [100,300,800], returns counts of peaks under 100, between 100 and 300, between 300 and 800, and above 800
    
    Args:
        peak_intensities (numpy array): peak intensities for a given histogram
        binned_intensities (numpy array): peak intensity intervals, within which to count the number of peaks

    Returns
        numpy array: the number of peaks between the specified binned intensities
    '''
    n_peaks_per_bin_size = []

    for i in range(len(binned_intensities)+1):
        if i == 0: # first bin size
            bin_start = -np.inf
            bin_end = binned_intensities[i]
        elif i == len(binned_intensities): # last bin size
            bin_start = bin_end
            bin_end = np.inf
        else: # any bin size in between
            bin_start = bin_end
            bin_end = binned_intensities[i]
        
        above_start = peak_intensities >= bin_start
        below_end = peak_intensities < bin_end

        in_between = above_start & below_end

        n_peaks_per_bin_size += [np.sum(in_between)]
    
    return n_peaks_per_bin_size

def get_fwhm_v2(peaks,crystalline_data, q_width):
    '''
    Get the full width at half maximum for each histogram peak.

    Args:
        peak_data (pandas DataFrame): peak starts, locations, and ends
        crystalline_data (numpy array): array of histogram intensities
        q_width (float): width of histograms in q-units

    Returns:
        array: full width half maximum values for each peak
    '''
    fwhm_array = np.zeros(len(peaks))

    for i in range(len(peaks)):
        peak = peaks[i]
        
        # get value of peak intensity
        peak_index = np.argmax(peak)
        peak_intensity = peak[peak_index]


        # get indices of peak edges closest to half the max intensity
        distances_from_half_max = np.abs(peak - peak_intensity/2)
        sorted_indices = np.argsort(distances_from_half_max)
        try:
            start = sorted_indices[(sorted_indices < peak_index)][0] # this index is w.r.t the peak ONLY not the entire plot
            end = sorted_indices[(sorted_indices > peak_index)][0] # this index is w.r.t the peak ONLY not the entire plot
        except:
            continue # some "stubby peaks" that already start with high intensities may have half intensities lower than the peak start/end intensities

        # calculate width and normalize to the width of the plot and q width
        fwhm = (end - start) / len(crystalline_data) / q_width
        fwhm_array[i] = fwhm
    
    return fwhm_array