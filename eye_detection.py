import cv2
import os



# Display the image with detected eyes
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def filter_eye(img):
  # Read the input image
  # img = # `cv2` is a module in Python's OpenCV library, which stands for Computer Vision. It
  # provides functions and tools for image processing, computer vision tasks, and machine
  # learning algorithms related to computer vision. In the provided code snippet, `cv2` is
  # used for reading images, converting images to grayscale, detecting eyes using Haar
  # cascades, and displaying images with detected eyes.
  # cv2.imread('dataset_copy/src-image/WhatsApp Image 2024-07-04 at 21.33.02.jpeg')

  # Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Read the Haar cascade for eye detection
  # eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
  eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

  # Detect eyes in the entire image
  eyes = eye_cascade.detectMultiScale(gray, 1.3, 4)
  print('Number of detected eyes:', len(eyes))

  for idx, (x, y, w, h) in enumerate(eyes):
      # Crop the eye region
      eye_img = img[y:y+h, x:x+w]
      
      return eye_img
      cv2.imshow('Eyes Detection', eye_img)
  return False