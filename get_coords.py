# get_coords.py
import cv2
import numpy as np

# --- CONFIGURATION ---
# Change this to the image you want to check
IMAGE_PATH = "field_map.png" 
# ---

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")
        # Draw a circle on the image to mark the point
        cv2.circle(params['image'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', params['image'])

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not load image at {IMAGE_PATH}")
else:
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event, {'image': img})
    print(f"Click on the image to get coordinates. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()