import numpy as np

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
                            'pct_area_under_raw_data': 1} 

feature_names = list(final_features_and_weights.keys())


def get_n_outliers(property_values,std_threshold=1.5):
    median = np.median(property_values)
    threshold = std_threshold * np.std(property_values)  
    n_outliers = np.sum(property_values - (median + threshold) > 0)
    return n_outliers

def get_peak_asymmetry(crystalline_data,peak_data,q_width): # this feature may be too hacky - not very interpretable for paper, better version may be floating peaks
    # Gets percent differences of intensities between peak beginnings and ends in order to find peaks that are asymmetric (relatively high peak start/end)

    peak_start_intensities = crystalline_data[peak_data['Peak Start Index']]
    peak_intensities = crystalline_data[peak_data['Peak Index']]
    peak_end_intensities = crystalline_data[peak_data['Peak End Index']]

    peak_intensity_difference = np.abs(peak_start_intensities - peak_end_intensities)
    peak_width_difference = np.abs(peak_data['Peak Start Index'] - peak_data['Peak End Index']) / len(crystalline_data) / q_width # need to normalize w.r.t total width of the plot AND the q_width

    # normalize differences w.r.t peak intensity
    normalized_slope = peak_intensity_difference / peak_width_difference / peak_intensities

    return normalized_slope

def get_n_floating_peaks(crystalline_data,peak_data,min_threshold = 50):
    peak_start_intensities = crystalline_data[peak_data['Peak Start Index']]
    peak_end_intensities = crystalline_data[peak_data['Peak End Index']]

    starts_above_threshold = peak_start_intensities > min_threshold
    ends_above_threshold = peak_end_intensities > min_threshold

    both_above_threshold = starts_above_threshold & ends_above_threshold
    return np.sum(both_above_threshold)

def get_peak_start_end_intensities(crystalline_data,peak_data):
    peak_start_intensities = crystalline_data[peak_data['Peak Start Index']]
    peak_end_intensities = crystalline_data[peak_data['Peak End Index']]

    averaged_intensities = np.mean(np.vstack([peak_start_intensities,peak_end_intensities]),axis=0)
    
    return averaged_intensities

def get_peak_base_widths(peak_data,crystalline_data,q_width):
    widths = peak_data['Peak End Index'] - peak_data['Peak Start Index']
    return widths / len(crystalline_data) / q_width # need to normalize by the total width of the plot AND the q_width

def get_max_mean_std(data):
    return np.max(data),np.mean(data),np.std(data)

def get_derived_features(data):
    pctile_25 = np.quantile(data,0.25)
    pctile_75 = np.quantile(data,0.75)
    mean = np.mean(data)
    max = np.max(data)

    return pctile_25, pctile_75, mean, max

def get_n_peaks_per_bin_size(peak_intensities,bin_sizes):
    # example: for bin_sizes = [100,300,800], returns counts of peaks under 100, between 100 and 300, between 300 and 800, and above 800
    n_peaks_per_bin_size = []

    for i in range(len(bin_sizes)+1):
        if i == 0: # first bin size
            bin_start = -np.inf
            bin_end = bin_sizes[i]
        elif i == len(bin_sizes): # last bin size
            bin_start = bin_end
            bin_end = np.inf
        else: # any bin size in between
            bin_start = bin_end
            bin_end = bin_sizes[i]
        
        above_start = peak_intensities >= bin_start
        below_end = peak_intensities < bin_end

        in_between = above_start & below_end

        n_peaks_per_bin_size += [np.sum(in_between)]
    
    return n_peaks_per_bin_size


def get_fwhm_v2(peaks,crystal_data, q_width):
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
        fwhm = (end - start) / len(crystal_data) / q_width
        fwhm_array[i] = fwhm
    
    return fwhm_array