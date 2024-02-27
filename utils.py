from pathlib import Path
import numpy as np
import tifffile
from skimage import measure, io
import os


def read_images(directory_path):
    """Reads all the images in the input path and organizes them according to the well_id"""
    # Define the directory containing your files
    directory_path = Path(directory_path)

    # Initialize a dictionary to store the grouped (per well) files
    images_per_well = {}

    # Iterate through the files in the directory
    for file_path in directory_path.glob("*"):
        # Check if the path is a file and ends with ".TIF"
        if file_path.is_file() and file_path.suffix.lower() == ".tif":
            # Get the filename without the extension
            filename = file_path.stem
            # Remove unwanted files (Plate_R files)
            if "Slide_M" in filename:
                pass
            # Remove maximum projections
            elif "_z" not in filename:
                pass
            else:
                # Extract the last part of the filename (e.g., A06f00d0)
                last_part = filename.split("_")[-1]

                # Get the first three letters to create the group name (well_id)
                well_id = last_part[:3]

                # Check if the well_id exists in the dictionary, if not, create a new list
                if well_id not in images_per_well:
                    images_per_well[well_id] = []

                # Append the file to the corresponding group
                images_per_well[well_id].append(str(file_path))

    return images_per_well


def min_intensity_projection(image_paths):
    """Takes a collection of image paths containing one z-stack per file and performs minimum intensity projection"""
    # Load images from the specified paths
    image_collection = io.ImageCollection(image_paths)
    # Stack images into a single 3D numpy array
    stack = io.concatenate_images(image_collection)
    # Perform minimum intensity projection along the z-axis (axis=0)
    min_proj = np.min(stack, axis=0)

    return min_proj


def save_min_projection_imgs(images_per_well, output_dir="./output/MIN_projections"):
    """Takes a images_per_well from read_images as input, performs minimum intensity projection and saves the resulting image on a per well basis"""
    for well_id, files in images_per_well.items():
        # Perform minimum intensity projection of the stack stored under well_id key
        min_proj = min_intensity_projection(images_per_well[well_id])

        # Create a directory to store the tif files if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Construct the output file path
        output_path = os.path.join(output_dir, f"{well_id}.tif")

        # Save the resulting minimum projection
        tifffile.imwrite(output_path, min_proj)
