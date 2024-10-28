import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Disable TensorFlow optimizations for simplicity
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/SC_Detaction_Model_First.keras"

# Load the pre-trained model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load class names
with open(f"{working_dir}/class_indices.json") as f:
    class_indices = json.load(f)

# Function to load and preprocess the image
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file)
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_file):
    preprocessed_img = load_and_preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices[str(predicted_class_index)]

# Streamlit App
st.title('Plant Disease Classifier')

# Upload an image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    with col2:
        if st.button('Classify'):
            model = load_model()
            if model:
                prediction = predict_image_class(model, uploaded_image)
                st.success(f'Prediction: {prediction}')
                import gc
                gc.collect()  # Explicitly collect garbage
