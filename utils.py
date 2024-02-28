from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import io, color, morphology
from skimage.measure import regionprops_table
import pyclesperanto_prototype as cle
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import pandas as pd
import os
import shutil

# Select a GPU with the following in the name. This will fallback to any other GPU if none with this name is found
cle.select_device("RTX")


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


def extract_labels(
    input_folder,
    top_hat_filter="scikit",
    filter_radius=20,
    object_detection="voronoi_otsu_nsbatwm",
    dilation_radius=2,
    erosion_radius=2,
    size_exclusion=True,
    exclusion_area=30,
    output_dir="./output/labels",
):

    # Define the directory containing your files
    directory_path = Path(input_folder)

    # Loop through all input images (minimum intensity projections)
    for image_path in directory_path.glob("*.tif"):

        # Load the image
        image = io.imread(image_path)

        # Get the filename without the extension
        filename = Path(image_path).stem

        # Convert to grayscale
        image_gray = color.rgb2gray(image)

        # Rescale to 0-255 and convert to uint8
        image_gray_uint8 = (image_gray * 255).astype(np.uint8)

        inverted_image = 255 - image_gray_uint8

        # Tophat filtering methods
        if top_hat_filter == "cle":
            result_image = None
            img_gpu = cle.push(inverted_image)
            result_image = cle.top_hat_box(
                img_gpu, result_image, radius_x=filter_radius, radius_y=filter_radius
            )
            result_image = cle.pull(result_image)

        elif top_hat_filter == "scikit":
            square_kernel = morphology.square(filter_radius)
            result_image = morphology.white_tophat(
                inverted_image, footprint=square_kernel
            )

        else:
            print("Choose either 'cle' or 'scikit' filtering methods")

        # Object detection methods (#TODO: implement APOC object_classifier if needed)

        if object_detection == "voronoi_otsu_nsbatwm":

            labels = nsbatwm.voronoi_otsu_labeling(
                result_image, spot_sigma=4, outline_sigma=1
            )

        exclude_border = nsbatwm.remove_labels_on_edges(labels)

        dilated_labels = cle.dilate_labels(exclude_border, radius=dilation_radius)

        eroded_labels = cle.erode_labels(dilated_labels, radius=erosion_radius)

        if size_exclusion:
            output_labels = cle.exclude_small_labels(
                eroded_labels, maximum_size=exclusion_area
            )
        else:
            output_labels = eroded_labels

        # Save the resulting labels image

        # Create a directory to store the tif files if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Construct the output file path
        output_path = os.path.join(output_dir, f"{filename}.tif")

        # Save the resulting minimum projection
        tifffile.imwrite(output_path, output_labels)


def extract_morphology_stats(input_folder):

    # Define the directory containing your files
    directory_path = Path(input_folder)

    # Initialize an empty list to store dataframes
    dataframes = []

    # Loop through all input images (minimum intensity projections)
    for image_path in directory_path.glob("*.tif"):

        # Load the image
        image = io.imread(image_path)

        # Get the filename without the extension
        filename = image_path.stem

        # Extract regionprops
        props = regionprops_table(
            label_image=image,
            properties=["label", "area_filled", "perimeter", "solidity"],
        )

        # Construct a dataframe
        df = pd.DataFrame(props)

        # Add the well_id column with the filename
        df.insert(0, "well_id", filename)

        # Append the dataframe to the list
        dataframes.append(df)

    # Concatenate all dataframes in the list into a single dataframe
    final_df = pd.concat(dataframes, ignore_index=True)

    return final_df


def copy_csv_results(results_directory):
    """Copy all .csv files from each plate folder into a per_organoid_stats folder under results_directory"""

    # Create an empty list to store the subdirectories within results_directory
    subdirectories = []

    # Iterate over subfolders in the results_directory and add them to subdirectories list
    for subfolder in results_directory.iterdir():
        if subfolder.is_dir():
            subdirectories.append(subfolder.name)

    # Create the destination folder to copy the .csv files contained in each subdir
    try:
        csv_results_path = os.path.join(results_directory, "per_organoid_stats")
        os.makedirs(csv_results_path)
    except FileExistsError:
        print(f"Directory already exists: {csv_results_path}")

    # Iterate over each subdirectory to scan and copy .csv files contained within
    for subdir in subdirectories:
        # Build the path to each of the subdirectories
        subdirectory_path = Path(results_directory) / subdir

        # Check if the subdirectory exists
        if subdirectory_path.exists():
            # Scan for .csv files in each subdirectory
            for file_path in subdirectory_path.glob("*.csv"):
                # Copy the file to the destination subfolder
                shutil.copy2(file_path, csv_results_path)


def extract_summary_stats(csv_path):
    """Processes a per_organoid .csv results file counting the number of occurrences and calculate the average of each property returning a summary_stats_df"""
    # Read the .csv into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Group by 'well_id' and calculate the maximum 'label' value and mean for other columns
    df_aggregated = (
        df.groupby("well_id")
        .agg(
            {
                "label": "max",  # Find the maximum value for 'label'
                "area_filled": "mean",
                "perimeter": "mean",
                "solidity": "mean",
            }
        )
        .reset_index()
    )

    # Rename 'label' to 'organoid_counts' and add '_avg' suffix to other columns
    df_aggregated = df_aggregated.rename(
        columns={
            "label": "organoid_counts",
            "area_filled": "area_filled_avg",
            "perimeter": "perimeter_avg",
            "solidity": "solidity_avg",
        }
    )

    # Reorder the columns if needed (assuming you want 'organoid_counts' right after 'well_id')
    df_aggregated = df_aggregated[
        [
            "well_id",
            "organoid_counts",
            "area_filled_avg",
            "perimeter_avg",
            "solidity_avg",
        ]
    ]

    # Extract the plate_name from the csv_path
    csv_path = Path(csv_path)
    plate_name = csv_path.stem

    # Adding the new column 'plate_name' to the left of df_merged
    df_aggregated.insert(0, "plate_name", plate_name)

    return df_aggregated


def save_summary_stats(results_directory):
    """Extracts summary stats from per object results and concatenates all df into a final_df (+save to disk as .csv) containing all plates data"""
    # Define the directory containing the copied per_organoid_stats
    csv_results_path = Path(os.path.join(results_directory, "per_organoid_stats"))

    # Initialize an empty DataFrame to collect all summary dataframes
    final_df = pd.DataFrame()

    # Scan for .csv files in each subdirectory
    for csv_path in csv_results_path.glob("*.csv"):
        summary_df = extract_summary_stats(csv_path)
        # Append the summary_df to the final_df
        final_df = pd.concat([final_df, summary_df], ignore_index=True)

    # Create the summary_stats directory if it doesn't exist
    summary_stats_directory = Path(results_directory, "summary_stats")
    summary_stats_directory.mkdir(parents=True, exist_ok=True)

    # Save the final_df as a .csv file under the summary_stats directory
    final_csv_path = summary_stats_directory / "summary_stats.csv"
    final_df.to_csv(final_csv_path, index=False)


def plot_plate(
    resolution, output_path, img_folder_path, show_fig=True, colormap="gray"
):
    """Plot images in a grid-like fashion according to the well_id position in the plate"""
    try:
        # Initialize a dictionary to store images by rows (letters)
        image_dict = {}

        # Iterate through the image files in the folder
        for filename in os.listdir(img_folder_path):
            if filename.endswith(".tif"):
                # Extract the first letter and the number from the filename
                first_letter = filename[0]
                number = int(filename[1:3])

                # Create a dictionary entry for the first letter if it doesn't exist
                if first_letter not in image_dict:
                    image_dict[first_letter] = {}

                # Create a dictionary entry for the number if it doesn't exist
                if number not in image_dict[first_letter]:
                    image_dict[first_letter][number] = []

                # Append the image filename to the corresponding number
                image_dict[first_letter][number].append(filename)

        # Sort the dictionary by keys (letters) and nested dictionary by keys (numbers)
        sorted_image_dict = {
            letter: dict(sorted(images.items()))
            for letter, images in sorted(image_dict.items())
        }

        # Calculate the number of rows based on the number of letters
        num_rows = len(sorted_image_dict)

        # Calculate the number of columns based on the maximum number
        num_cols = max(max(images.keys()) for images in sorted_image_dict.values())

        # Calculate the figsize based on the number of columns and rows
        fig_width = num_cols * 3  # Adjust the multiplier as needed
        fig_height = num_rows * 2.5  # Adjust the multiplier as needed

        # Create a subplot for each image, using None for empty subplots
        if num_rows == 1:
            fig, axes = plt.subplots(
                1, num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True
            )
        elif num_cols == 1:
            fig, axes = plt.subplots(
                num_rows, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True
            )
        else:
            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(fig_width, fig_height),
                sharex=True,
                sharey=True,
            )

        for i, (letter, images) in enumerate(sorted_image_dict.items()):
            for j, (number, filenames) in enumerate(images.items()):
                if filenames:
                    image_filename = filenames[0]  # Use the first filename in the list
                    image_path = os.path.join(img_folder_path, image_filename)
                    image = tifffile.imread(image_path, is_ome=False)

                    if num_rows == 1:
                        axes[j].imshow(image, cmap=colormap)
                        axes[j].set_title(f"{letter}{number:02d}")
                        axes[j].axis("off")
                    elif num_cols == 1:
                        axes[i].imshow(image, cmap=colormap)
                        axes[i].set_title(f"{letter}{number:02d}")
                        axes[i].axis("off")
                    else:
                        axes[i, j].imshow(image, cmap=colormap)
                        axes[i, j].set_title(f"{letter}{number:02d}")
                        axes[i, j].axis("off")

                else:
                    # If there are no images for a specific letter-number combination, remove the empty subplot
                    if num_rows == 1:
                        fig.delaxes(axes[j])
                    elif num_cols == 1:
                        fig.delaxes(axes[i])
                    else:
                        fig.delaxes(axes[i, j])

        # Adjust the spacing and set aspect ratio to be equal
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # Save the plot at a higher resolution
        plt.savefig(output_path, format="tif", dpi=resolution, bbox_inches="tight")

        # If False plt.show() is not run to avoid loop stop upon display
        if show_fig:
            # Show the plot (optional)
            plt.show()

    except TypeError:
        print(f"You can find your individual minimum projections under {output_path}")
