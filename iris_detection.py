import cv2
import numpy as np

# Import xml files for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

kernel = np.ones((3, 3), np.uint8)  # Kernel for morphology

# Load the image
frame = cv2.imread("dataset_copy/src-image/WhatsApp Image 2024-07-04 at 21.33.01.jpeg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect eyes
detect_eye = eye_cascade.detectMultiScale(gray, 1.3, 4)

print("Eyes detected:", len(detect_eye))

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

for (eye_x, eye_y, eye_w, eye_h) in detect_eye:
    eye_region = gray[eye_y:eye_y + eye_h, eye_x:eye_x + eye_w]

    # Crop the center of the eye for processing
    eye_cropped = crop_center(eye_region)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(eye_cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to clean the binary image
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel)

    # Find contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour assuming it's the iris
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the moments for the largest contour
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            Cx = int(M['m10'] / M['m00'])
            Cy = int(M['m01'] / M['m00'])
            
            # Adjust coordinates to the original image
            center = (Cx + eye_x + (eye_w - eye_cropped.shape[1]) // 2, Cy + eye_y + (eye_h - eye_cropped.shape[0]) // 2)
            
            # Draw the detected iris center
            cv2.circle(frame, center, 3, (0, 255, 0), 2)
        else:
            print("Moment calculation failed, contour area too small.")
    else:
        print("No contours found for the eye region.")

    # Display intermediate results for debugging
    cv2.imshow('Eye Region', eye_region)
    cv2.imshow('Eye Cropped', eye_cropped)
    cv2.imshow('Binary Image', binary)
    cv2.imshow('Dilated Image', dilate)

# Show the result
cv2.imshow('Iris Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
