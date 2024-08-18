import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split       
def change_color_outside_circle(image, color=(0, 0, 0)):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Calculate the radius of the circle (17% of the width)
    radius = int(0.17 * width)

    # Create a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, thickness=-1)

    # Create a color image for the background
    background = np.full_like(image, color)

    # Combine the background and the original image using the mask
    result = np.where(mask[..., np.newaxis] == 255, image, background)

    return result




def filter_eye(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 4)
    print('Number of detected eyes:', len(eyes))

    for idx, (x, y, w, h) in enumerate(eyes):
        eye_img = img[y:y+h, x:x+w]
        
        # Convert the eye region to grayscale and blur it
        eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        eye_blurred = cv2.medianBlur(eye_gray, 5)

        # Detect the lens (assumed to be the largest circle in the eye region)
        circles = cv2.HoughCircles(eye_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                   param1=100, param2=30, minRadius=20, maxRadius=100)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                # Calculate the new bounding box for the eye centered on the lens
                center_x, center_y = cx, cy
                crop_size = min(w, h)  # Use the smaller dimension to ensure the crop stays within bounds

                # Define the new bounding box centered on the lens
                new_x = max(center_x - crop_size // 2, 0)
                new_y = max(center_y - crop_size // 2, 0)
                new_x_end = min(new_x + crop_size, eye_img.shape[1])
                new_y_end = min(new_y + crop_size, eye_img.shape[0])

                # Crop the image centered on the lens
                centered_eye_img = eye_img[new_y:new_y_end, new_x:new_x_end]
                return centered_eye_img
            
        else:
            print("No circle detected")

    return None


def crop_center(image, crop_width=80, crop_height=80):
    # Get image dimensions
    image_height, image_width = image.shape[:2]

    # Calculate center point
    center_x, center_y = image_width // 2, image_height // 2

    # Calculate the starting and ending coordinates for the crop
    start_x = max(center_x - crop_width // 2, 0)
    end_x = min(center_x + crop_width // 2, image_width)
    start_y = max(center_y - crop_height // 2, 0)
    end_y = min(center_y + crop_height // 2, image_height)

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    return cropped_image