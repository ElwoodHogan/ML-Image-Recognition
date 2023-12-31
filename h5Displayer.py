import h5py
import cv2
import numpy as np

FILE_TO_DISPLAY = "dataset700.h5"
TEXT_FILE_TO_SAVE_TO = "Test.txt"

def display_dataset(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        # Loop through each dataset: train, test
        for dataset_name in ["train", "test"]:
            images = hf[f"{dataset_name}_images"]
            labels = hf[f"{dataset_name}_labels"]

            for i in range(len(images)):
                img = images[i]
                label = labels[i]
                
                # Display image using OpenCV
                cv2.imshow(f"{dataset_name} - Label: {label}", img)
                key = cv2.waitKey(0) # Wait until a key is pressed

                # Close window if 'esc' key is pressed
                if key == 27: 
                    cv2.destroyAllWindows()
                    return

                # Close the current window to show the next image
                cv2.destroyAllWindows()

def display_dataset_info(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        # Loop through each dataset: train, test
        for dataset_name in ["train", "test"]:
            images = hf[f"{dataset_name}_images"]
            labels = hf[f"{dataset_name}_labels"]

            # As the images are stored as raw data, the paths need to be reconstructed. 
            # If the original paths were stored in the HDF5, this step would not be necessary.
            # For the purpose of this example, we'll assume the paths were not stored 
            # and instead generate a placeholder path based on index and dataset name.
            
            for i in range(len(images)):
                img_path = f"{dataset_name}_img_{i}.tif"  # Placeholder path
                label = labels[i]
                
                print(f"Image: {img_path} | Label: {label} | Set: {dataset_name}")

def display_dataset_info_to_file(h5_path, output_file):
    with h5py.File(h5_path, 'r') as hf, open(output_file, 'w') as out_file:
        for dataset_name in ["train", "test"]:
            images = hf[f"{dataset_name}_images"]
            labels = hf[f"{dataset_name}_labels"]
            
            for i in range(len(images)):
                img_path = f"{dataset_name}_img_{i}.tif"  # Placeholder path
                label = labels[i]
                
                out_file.write(f"Image: {img_path} | Label: {label} | Set: {dataset_name}\n")

if __name__ == "__main__":
    h5_path = FILE_TO_DISPLAY
    display_dataset_info_to_file(h5_path, TEXT_FILE_TO_SAVE_TO)