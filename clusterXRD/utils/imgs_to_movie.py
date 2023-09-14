from PIL import Image
import glob
import numpy as np
import os
import shutil

def make_gif(frame_folder,
             output_name,
             gif_dir,
             n_images=20,
             n_duration=300):
    '''
    Converts all images in a directory into a .gif movie

    Args:
        frame_folder (str): path to directory containing all images of interest
        output_name (str): gif filename without the .gif extension
        gif_dir (str): directory to place saved .gif files into
        n_images (int): max number of images in the gif, will random sample from all images if < total number of images in frame_folder
        n_duration (int): duration of each frame in ms

    Returns:
        None: saves .gif files locally
    '''
    
    all_frame_paths = glob.glob(f"{frame_folder}/*.png")

    if len(all_frame_paths) == 0: return None
    if len(all_frame_paths) < n_images: n_images = len(all_frame_paths)
    frame_paths = np.random.choice(all_frame_paths,size=n_images,replace=False)
    frames = [Image.open(image) for image in frame_paths]
    
    frame_one = frames[0]
    frame_one.save(f'{gif_dir}/{output_name}.gif', format="GIF", append_images=frames,
               save_all=True, duration=n_duration, loop=0)
