import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# Function to preprocess the image
def preprocess_image(image_path, img_rows=224, img_cols=224):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        exit(1)

    # Resize the image
    image = cv2.resize(image, (img_rows, img_cols))

    # Convert the image to an array and normalize it
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image /= 255

    return image


# Load the trained model
model = load_model('eyedisease1.h5')

# Dictionary for class labels
class_labels = ['Bulging_Eyes', 'Closed', 'Crossed_Eyes', 'Glaucoma', 'Uveitis']

# Image path for testing
image_path = "C:/Users/USER/Downloads/image-3.jpeg"

# Preprocess the image
image = preprocess_image(image_path)

# Make a prediction
prediction = model.predict(image)
predicted_class = np.argmax(prediction, axis=1)

# Print the result
print(f"The image is predicted to be: {class_labels[predicted_class[0]]}")
