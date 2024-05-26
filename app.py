import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("cnn.h5")

# Define class labels
map_dict = {
   'Arive-Dantu': 0,
 'Basale': 1,
 'Betel': 2,
 'Crape_Jasmine': 3,
 'Curry': 4,
 'Drumstick': 5,
 'Fenugreek': 6,
 'Guava': 7,
 'Hibiscus': 8,
 'Indian_Beech': 9,
 'Indian_Mustard': 10,
 'Jackfruit': 11,
 'Jamaica_Cherry-Gasagase': 12,
 'Jamun': 13,
 'Jasmine': 14,
 'Karanda': 15,
 'Lemon': 16,
 'Mango': 17,
 'Mexican_Mint': 18,
 'Mint': 19,
 'Neem': 20,
 'Oleander': 21,
 'Parijata': 22,
 'Peepal': 23,
 'Pomegranate': 24,
 'Rasna': 25,
 'Rose_apple': 26,
 'Roxburgh_fig': 27,
 'Sandalwood': 28,
 'Tulsi': 29
}

st.title("Plant Species Recognition")

# File upload section
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (64, 64))  # Adjust the size as needed
    
    # Display the uploaded image
    st.image(opencv_image, caption="Uploaded Image", use_column_width=True)

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]

    # Generate prediction button
    if st.button("Generate Prediction"):
        prediction = model.predict(img_reshape).argmax()
        predicted_label = list(map_dict.keys())[list(map_dict.values()).index(prediction)]
        st.write(f"Predicted Label for the image is {predicted_label}")
