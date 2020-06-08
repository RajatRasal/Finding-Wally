"""
How to make dataset:
For each candidate region tag it with its original image
Identify whether it is a foreground or background based on IOU with real image,
threshold > 0.2
For each foreground, calculate an offset to the ground truth
From each image we want a 25% foreground and 75% background
Reshape each region proposal to 224 * 224
"""
nos = [1, 2, 4, 5, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27]

def create_datapoints(full_img, target_region_bbox, proposal_bbox, t=0.2):
+++
    return 

"""
Network:
Keep backbone of VGG16 and make it untrainable.
Add 2 more layers FC layers to it and make those trainable.
Add 2 separable channels of outputs - one for MSE another for BCE.
Use a multitask loss function where we only include the MSE loss if the FG = 1.
"""

"""
Training:
Using ImageDataGenerator
1) Mean centered input
2) Width shift + height shift 2 pixels in either direction
"""

"""
Results:
For all those which were predicted foreground, apply the offset and accumulate.
NMS over accumulator / thresholding.
Draw a box.
"""
"""
Produce a dataset
"""
