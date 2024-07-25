import warnings
warnings.filterwarnings('ignore')

# # import cv2
# import numpy as np
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing import image
#
# model = ResNet50(weights='imagenet')
#
#
# def load_and_preprocess_image(image_path):
#     img = image.load_img(image_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array
#
#
# def detect_faces(image_path):
#     img_array = load_and_preprocess_image(image_path)
#     features = model.predict(img_array)
#
#     threshold = 0.5
#     if np.mean(features) > threshold:
#         return "Face Detected", np.mean(features)
#     else:
#         return "No Face Detected", np.mean(features)
#
#
# if __name__ == '__main__':
#     test_image = r"100k-ai-faces-5.jpg"
#     result_label, confidence = detect_faces(test_image)
#
#     print(f"Result: {result_label}")
#     print(f"Confidence: {confidence}")

import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load pre-trained ResNet50 model without the top (classification) layer
model = ResNet50(weights='imagenet', include_top=False)

# Load and preprocess image
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# Face detection function
def detect_faces(image_path):
    img_array = load_and_preprocess_image(image_path)

    # Use the pre-trained ResNet50 model to extract features
    features = model.predict(img_array)

    # Use OpenCV for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        return "Face Detected", np.mean(features)
    else:
        return "No Face Detected", np.mean(features)


# Example of face detection
image_path = r"100k-ai-faces-5.jpg"
result_label, confidence = detect_faces(image_path)

print(f"Result: {result_label}")
print(f"Confidence: {confidence}")
