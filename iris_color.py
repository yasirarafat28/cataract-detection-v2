import cv2
import dlib
import numpy as np

# Load the face detector and the facial landmarks predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

def detect_iris(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    iris_coords = []
    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        # Detect iris using HoughCircles
        for eye_pts in [left_eye_pts, right_eye_pts]:
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [eye_pts], 255)
            eye = cv2.bitwise_and(gray, gray, mask=mask)
            eye = cv2.GaussianBlur(eye, (9, 9), 2)
            circles = cv2.HoughCircles(eye, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=30)
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    radius = i[2]
                    iris_coords.append((center, radius, eye_pts))
                    
    return iris_coords

def change_iris_color(image, iris_coords, color):
    output = image.copy()
    for (center, radius, eye_pts) in iris_coords:
        mask = np.zeros_like(image)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        output[mask == 255] = color
    return output

def crop_iris(image, iris_coords):
    cropped_images = []
    for (center, radius, eye_pts) in iris_coords:
        x, y = center
        r = radius
        cropped_image = image[y-r:y+r, x-r:x+r]
        cropped_images.append(cropped_image)
    return cropped_images

def main():
    # Load the image
    image = cv2.imread("dataset_copy/src-image/nahid.jpg")
    if image is None:
        print("Error: Could not load image. Please check the file path.")
        return

    # Detect the iris
    iris_coords = detect_iris(image)
    if not iris_coords:
        print("Error: No iris detected. Please use a clearer image.")
        return

    # Change the iris color to green (for example)
    colored_image = change_iris_color(image, iris_coords, (0, 255, 0))

    # Crop the iris images
    cropped_iris_images = crop_iris(image, iris_coords)

    # Save the results
    cv2.imwrite("colored_eye_image.jpg", colored_image)
    for i, cropped_image in enumerate(cropped_iris_images):
        cv2.imwrite(f"cropped_iris_{i}.jpg", cropped_image)

    # Display the results
    cv2.imshow("Colored Eye Image", colored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
