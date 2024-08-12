import cv2
import dlib

print(dlib.__version__)
print(cv2.__version__)
# Load the face detector from dlib
detector = dlib.get_frontal_face_detector()

def main():
    # Load the image
    image_path = "dataset_copy/src-image/nahid.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Please check the file path.")
        return

    print("Image loaded successfully")
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Gray image shape:", gray.shape)
    print("Gray image dtype:", gray.dtype)

    # Detect faces
    try:
        faces = detector(gray)
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        return

    if len(faces) == 0:
        print("No faces detected.")
        return

    print(f"Number of faces detected: {len(faces)}")

    # Draw rectangles around detected faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the results
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
