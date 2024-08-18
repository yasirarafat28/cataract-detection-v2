import cv2
import numpy as np

from utils import filter_eye


# Example usage:
img = cv2.imread('dataset_copy/src-image/WhatsApp Image 2024-07-04 at 21.29.50.jpeg')
centered_eye_img = filter_eye(img)
if centered_eye_img is not None:
    cv2.imshow('Centered Eye', centered_eye_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
