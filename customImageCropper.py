'''
Copyright (c) 2020 Hao Da (Kevin) Dong
@file       customImageCropper.py
@date       2020/04/02
@brief      Application for user to crop an image using arbitrary mouse clicks
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2

# Mouse event callback function
def recordClick(event, x, y, flags, param):
    # Grab references to global variables
    global mouseClicks
 
    # If there is a left click, record the location
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseClicks.append([x,y])


# Read in image to crop
image = cv2.imread('dataset/images/um_000001.png')

# Create window to detect mouse events
cv2.namedWindow("Image", flags=cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", recordClick)

# List to record mouse clicks
mouseClicks = []

while (True):
    cv2.imshow('Image', image)
 
    key = cv2.waitKey(1) & 0xFF

	# Exit loop if user presses 'q'
    if key == ord("q"):
        break

# Turn mouseClick into a NumPy array
mouseClicks = np.array(mouseClicks)

# Create binary mask with polygon created from mouse clicks
mask = np.zeros(image.shape, dtype=np.uint8)
cv2.fillPoly(mask, [mouseClicks], color=(255,255,255))

# Extract image region inside polygon and crop to the minimum bounding rectangle
croppedImage = cv2.bitwise_and(src1=image, src2=mask)
x, y, w, h = cv2.boundingRect(mouseClicks) 
croppedImage = croppedImage[y : y+h, x : x+w]

# Display cropped image
cv2.imshow('Cropped Image', croppedImage)

# Save image if user presses 's'
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    cv2.imwrite('training_set/orange.png', croppedImage)

print(mouseClicks)

