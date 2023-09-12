import numpy as np
import pandas as pd
from functools import reduce
from scipy.integrate import simpson

def remove_stubby_peaks(histogram_intensities,peak_data,min_peak_percentage=0.01):
    '''
    This function filters out peaks that are small (stubby) enough to be ignored.

    Args:
        histogram_intensities (Numpy array): array of crystalline histogram intensities
        peak_data (Pandas DataFrame): dataframe of peak start, location, and end indices
        min_peak_percentage (float), default 0.01: proportion of the pattern's maximum peak intensity, under which a peak should be ignored
    
    Returns:
        Pandas DataFrame: dataframe of peak start, location, and end indices, excluding stubby peaks 
    '''

    try:
        peak_start_intensities = histogram_intensities[peak_data['Peak Start Index']]
        peak_end_intensities = histogram_intensities[peak_data['Peak End Index']]

        min_start_end_intensity = np.min(np.stack([peak_start_intensities,peak_end_intensities]),axis=0)
        peak_intensity = histogram_intensities[peak_data['Peak Index']]
        peak_height = peak_intensity - min_start_end_intensity

        max_peak_height = np.max(peak_height)

        non_stubby_peaks = peak_data[peak_height > min_peak_percentage * max_peak_height]

        return non_stubby_peaks 
    except:
        return peak_data

def is_amorphous(crystalline_data,background_data,peak_data,max_peak_intensity_this_wafer):
    '''
    Detects whether a histogram belongs to an amorphous sample.

    Args:
        crystalline_data (Numpy array): array of crystalline histogram intensities
        background_data (Numpy array): array of background histogram intensities
        peak_data (Pandas DataFrame): dataframe of peak start, location, and end indices

    Returns:
        boolean: True for an amorphous sample

    '''
    peak_intensities = crystalline_data[peak_data['Peak Index']]
    amorphous_intensities_at_peak_positions = background_data[peak_data['Peak Index']]
    max_peak_intensity = np.max(crystalline_data)

    min_peak_threshold = 0.03 * np.max(crystalline_data)

    peak_intensities_above_min = peak_intensities[peak_intensities > min_peak_threshold]
    amorphous_intensities_at_peak_positions_above_min = amorphous_intensities_at_peak_positions[peak_intensities > min_peak_threshold]

    # peak_greater_than_background = peak_intensities_above_min > amorphous_intensities_at_peak_positions_above_min

    # calculate the percentage of peaks that are above amorphous intensities
    #pct_peak_greater = np.sum(peak_greater_than_background)/len(peak_intensities_above_min)

    # calculate the ratio of max crystal intensity to the max background intensity
    ratio_max_crystal_to_background = np.max(crystalline_data) / np.max(background_data)

    # calculate the percentage of crystalline data
    percent_area = get_percentage_area_under_raw_data(crystalline_data + background_data)

    if ratio_max_crystal_to_background > 5 and percent_area < 0.15 and max_peak_intensity / max_peak_intensity_this_wafer > 0.2: return False
    else: return True

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

def smooth_data(histogram_intensities,smooth_radius = 5):
    '''
    Take the rolling average of intensity values over a defined radius

    Args:
        histogram_intensities (numpy array): array of histogram intensities
        smooth_radius (int): number of indices before and after a given value for which to smooth intensities over

    Returns:
        numpy array: array of histograms intensities after smoothing
    '''
    intensities = []
    coeffs = []
    for shift in range(-smooth_radius,smooth_radius+1):
        weight_coeff = 1-np.abs(shift)/smooth_radius
        coeffs += [weight_coeff]
        intensities += [weight_coeff*np.roll(histogram_intensities,shift)]
    return np.sum(intensities,axis=0) / np.sum(coeffs)

def euclidean(array_1,array_2):

    difference = array_1 - array_2
    difference_squared = difference **2

    summed_difference_squared = np.sum(difference_squared,axis=1)

    return summed_difference_squared ** 0.5


def get_line_array_from_two_points(point1,point2,n=100):
    x1, y1 = point1
    x2, y2 = point2
    
    x_values = np.linspace(x1,x2,n)
    y_values = np.linspace(y1,y2,n)

    return np.vstack([x_values,y_values]).T

def get_best_offset(histogram_array, background_array, n_array_len = 50, max_offset = 30):
    offsets = np.linspace(0,max_offset,n_array_len)

    background_tiled = np.broadcast_to(background_array,[len(offsets),len(background_array)])
    offsets_tiled = offsets.reshape([len(offsets),1])
    background_offsets_tiled = background_tiled - offsets_tiled

    raw_tiled = np.broadcast_to(histogram_array,[len(offsets),len(histogram_array)])
    diff_tiled = background_offsets_tiled - raw_tiled

    summed_number_positive_diff = np.sum(diff_tiled > 0,axis=1)

    combined_line_values = get_line_array_from_two_points([offsets[0],summed_number_positive_diff[0]],[offsets[-1],summed_number_positive_diff[-1]],n=n_array_len)
    combined_sum_values = np.vstack([offsets,summed_number_positive_diff]).T
    distances = euclidean(combined_line_values,combined_sum_values)
    elbow_loc = combined_sum_values[np.argwhere(distances == np.max(distances))[0][0]]
    #print(f'Best offset is {elbow_loc[0]}')

    return elbow_loc[0]
