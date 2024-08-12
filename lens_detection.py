import cv2
import numpy as np

# Define HSV color ranges for eyes colors
class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
EyeColor = {
    class_name[0]: ((166, 21, 50), (240, 100, 85)),
    class_name[1]: ((166, 2, 25), (300, 20, 75)),
    class_name[2]: ((2, 20, 20), (40, 100, 60)),
    class_name[3]: ((20, 3, 30), (65, 60, 60)),
    class_name[4]: ((0, 10, 5), (40, 40, 25)),
    class_name[5]: ((60, 21, 50), (165, 100, 85)),
    class_name[6]: ((60, 2, 25), (165, 20, 65))
}

def check_color(hsv, color):
    return (color[0][0] <= hsv[0] <= color[1][0] and
            color[0][1] <= hsv[1] <= color[1][1] and
            color[0][2] <= hsv[2] <= color[1][2])

def find_class(hsv):
    color_id = 7
    for i in range(len(class_name) - 1):
        if check_color(hsv, EyeColor[class_name[i]]):
            color_id = i
            break
    return color_id

def eye_color(image):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0:2]
    
    eye_radius = min(h, w) // 4  # Approximate the radius of the pupil/lens
   
    # Create a mask for the eye area
    eye_center = (w // 2, h // 2)
    imgMask = np.zeros((h, w, 1), dtype=np.uint8)
    cv2.circle(imgMask, eye_center, int(eye_radius), (255), -1)

    eye_class = np.zeros(len(class_name), np.float32)

    for y in range(h):
        for x in range(w):
            if imgMask[y, x] != 0:
                eye_class[find_class(imgHSV[y, x])] += 1

    main_color_index = np.argmax(eye_class[:len(eye_class) - 1])
    total_vote = eye_class.sum()

    print("\n\nDominant Eye Color: ", class_name[main_color_index])
    print("\n **Eyes Color Percentage **")
    for i in range(len(class_name)):
        print(class_name[i], ": ", round(eye_class[i] / total_vote * 100, 2), "%")
    
    label = 'Dominant Eye Color: %s' % class_name[main_color_index]  
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (155, 255, 0), 2)
    cv2.imshow('EYE-COLOR-DETECTION', image)

if __name__ == '__main__':
    # Directly specify the eye image path
    image_path = 'image-to-test/No Cataract/IMG20230611193931.jpg'
    
    # Read the eye image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Detect eye color percentage
    eye_color(image)
    
    # Save the result
    cv2.imwrite('sample/result_eye_color.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
