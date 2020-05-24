"""
End-to-end object detection pipeline
"""
from PIL import Image

from model import load_model
from selective_search import (scale_image_in_aspect_ratio
    find_candidates
)


# Load model
saved_model_path = './saved_model'
model = load_model(saved_model_path)

# Run selective search on image
image_path = './data/original-images/19.jpg'
image = Image.open(image_path)

# Scale image to reduce no. of region proposals
width = 1000
image_scaled = scale_image_in_aspect_ratio(image, width)
height = image_scaled.shape[0]

assert image_scaled.shape[1] == 1000

# Region proposal using selective search
# TODO: Try tensorflow generate bounding box proposals
candidates = find_candidates(image)

# Normalise candidate box coordindates and put 
#  into tf.image required format -> [y1, x1, y2, x2]
norm_candidates = np.zeros((len(candidates), 4), dtype=np.float32)
for i in range(5):  # len(candidates)):
    # Plot using plt
    y1 = candidates[i].y / height
    x1 = candidates[i].x / width
    y2 = (y1 + candidates[i].h) / height
    x2 = (x1 + candidates[i].w) / width
    norm_candidates[i] = [y1, x1, y2, x2]
    # Plot using tf 

assert (norm_candidates == 0).sum() == 0

# Preprocess image - region proposal normalising, vgg_preprocessor
# Apply bbox to image - crop_and_resize
# convert types
# resize
# vgg.preprocessor

# Pass each image into model

# Get predictions - select highest regions of confidence (nms)
#  + unmangle bounding box prediction

# Draw boxes onto image 
