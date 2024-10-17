import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import distance_transform_edt
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Import classes
from soft_morph import SoftSkeletonizer, SoftErosion, SoftDilation, SoftClosing, SoftOpening


# Function to import the data and convert to tensor
def import_data(image_path):
    image_np = np.ceil(np.array(Image.open(image_path).convert("L"), dtype = np.int16)/255)
    image_ts = torch.tensor(image_np, device=DEVICE).unsqueeze(0).unsqueeze(0)
    return image_ts

# Function 
def generate_probabilistic_image(binary_image, scaling_factor):
    distances_fg = distance_transform_edt(binary_image) *scaling_factor
    distances_bg = distance_transform_edt(1 - binary_image)*scaling_factor
    distance_map = distances_fg - distances_bg
    min_val = np.min(distance_map)
    max_val = np.max(distance_map)
    probabilistic_image = (distance_map - min_val) / (max_val - min_val)
    return torch.tensor(probabilistic_image, dtype=torch.float32,device=DEVICE), distance_map

def makefigtable(listoutputs, row_titles, col_titles):
    nbrows, nbcols = len(listoutputs), len(listoutputs[0])
    # print(nbrows, nbcols)
    fig, axes = plt.subplots(nrows=nbrows, ncols=nbcols, figsize=(12, 5), sharex=True, sharey=True)

    for i in range(nbrows):
        # print(i)
        for j in range(nbcols):
            # print(j)
            axes[i, j].imshow(listoutputs[i][j], cmap=plt.cm.gray)
            axes[i, j].axis('off')

            # Set row titles
            if j == 0:
                # axes[i, j].set_ylabel(row_titles[i], fontsize=16, rotation=-90, ha='left', va='center')
                axes[i, j].text(-0.2, 0.5, row_titles[i], fontsize=15, rotation=90, ha='center', va='center', transform=axes[i, j].transAxes)

            # Set column titles
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=16)
            
    fig.tight_layout()
    plt.show()
    
    
def simple_operation(image):
    # Define filters
    erode = SoftErosion()
    erode.to(DEVICE)
    dilate = SoftDilation()
    dilate.to(DEVICE)
    skeleton = SoftSkeletonizer(5)
    skeleton.to(DEVICE)
    close = SoftClosing()
    close.to(DEVICE)
    openn = SoftOpening()
    openn.to(DEVICE)
    
    eroded = erode(image, 1,8)
    dilated = dilate(image, 2, 4)
    skeletonized = skeleton(image)
    closed = close(image, 2, 8, 8)
    opened = openn(image, 2, 4, 4)
    return [image.squeeze(), eroded.squeeze(), dilated.squeeze(), skeletonized.squeeze(), closed.squeeze(), opened.squeeze()]

def main():
    # Import data
    binary_image = import_data("data/example_data.png") # Import binary image
    probabilistic_image, dist_map = generate_probabilistic_image(binary_image, 0.05) # Create image with continuous values between 0 and 1
    
    # simple operation
    binary_results = simple_operation(binary_image)
    proba_results = simple_operation(probabilistic_image)
    col = ["Image", "Erosion", "Dilation", "Skeletonization", "Closing", "Opening"]
    row = ["Binary", "Probabilistic"]
    makefigtable([binary_results, proba_results],row, col)
    



main()