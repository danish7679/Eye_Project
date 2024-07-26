import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def preprocess_image(uploaded_file, img_rows=224, img_cols=224):
    # Get file content as bytes
    file_bytes = uploaded_file.getvalue()

    # Convert bytes to numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)

    # Decode numpy array as image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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

# Streamlit App
st.title('Eye Disease Prediction')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    image = preprocess_image(uploaded_file)
    if image is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        if st.button('Detect'):
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)
            st.success(f"The image is predicted to be: {class_labels[predicted_class[0]]}")
