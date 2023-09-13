import numpy as np
from functools import reduce
import pandas as pd
from .featurization import *

def find_peaks(histogram_intensities,histogram_filename,save_dir,stubby_peak_removal=True): 
        '''
        Identifies peak starts, locations, and ends for a given histogram intensity array.

        Args:
            histogram_intensities (numpy array): array of histogram intensities
            histogram_filename (str): filename for the histogram data from which peaks are identified
            save_dir (str): directory in which peak information will be saved, if None then nothing will be saved
            stubby_peak_removal (bool), default 'True': if True, will remove small/stubby peaks
        Returns:
            pandas dataframe: peak starts, locations, and ends
    
        '''
        n_datapoints = len(histogram_intensities)

        roll_index_lengths = [1,round(n_datapoints/1000),round(n_datapoints/200)] # maximum roll index length cannot be longer than half the width of the skinniest peak you wish to detect (measured in index units)

        peak_indices_list = []
        peak_starts_list = []
        peak_ends_list = []
        valley_indices_list = []

        for roll_index_length in roll_index_lengths:
            A = histogram_intensities 
            B = np.roll(histogram_intensities,roll_index_length) 
            C = A - B # change from previous point
            D = np.roll(C,-roll_index_length) # change to the next point

            E = C > 0 
            F = D > 0 
            G = C < 0 
            H = D < 0 
            I = C == 0 
            J = D == 0
            
            peak_indices_list += [np.argwhere(E & H).flatten()]
            peak_starts_list += [np.argwhere(I & F).flatten()]
            peak_ends_list += [np.argwhere(G & J).flatten()]
            valley_indices_list += [np.argwhere(G & F).flatten()]
        
        all_peak_indices = reduce(np.intersect1d,peak_indices_list) # find intersection between all peak indices lists
        all_peak_starts = peak_starts_list[0]
        all_peak_ends = peak_ends_list[0]
        all_valley_indices = valley_indices_list[0]

        # add all valleys to starts and ends since one valley = one start + one end

        all_peak_starts = np.sort(np.concatenate([all_peak_starts,all_valley_indices]))
        all_peak_ends = np.sort(np.concatenate([all_peak_ends,all_valley_indices]))
        
        max_number_peaks = len(all_peak_indices)

        peak_start_indices = np.zeros(max_number_peaks,dtype=int)
        peak_indices = np.zeros(max_number_peaks,dtype=int)
        peak_end_indices = np.zeros(max_number_peaks,dtype=int)
        
        peak_group_number = 0
        
        for i in all_peak_indices: # loop through each peak and find suitable corresponding start and end indices
            
            potential_starts = np.flip(all_peak_starts[all_peak_starts < i]) # need to flip because potential starts are evaluated from high values to low!
            potential_ends = all_peak_ends[all_peak_ends > i]

            start = 0  #initialize default value in case no corresponding start can be found
            end = 0
            
            try: start = potential_starts[0]
            except: pass

            try: end = potential_ends[0]
            except: pass

            peak_start_indices[peak_group_number] = start
            peak_indices[peak_group_number] = i
            peak_end_indices[peak_group_number] = end
            peak_group_number += 1
        
        df = pd.DataFrame(data={'Peak Start Index': peak_start_indices,
                                'Peak Index': peak_indices,
                                'Peak End Index': peak_end_indices},dtype=int)

        df = df[~(df == 0).any(axis=1)] # gets rid of any incomplete / fake peaks

        if stubby_peak_removal == True: df = remove_stubby_peaks(histogram_intensities,df)

        if save_dir: df.to_csv(f'{save_dir}/{histogram_filename}_peaks.csv')

        return df

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
