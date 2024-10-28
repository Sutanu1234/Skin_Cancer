import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Disable TensorFlow optimizations for simplicity
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}SC_Detaction_Model_First.keras"

# Load the pre-trained model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Load class names
with open(f"{working_dir}/class_indices.json") as f:
    class_indices = json.load(f)

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    # Open the image using Pillow
    img = Image.open(image_file)
    
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert to a numpy array and normalize
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_file, class_indices):
    preprocessed_img = load_and_preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title('Plant Disease Classifier')

# Upload an image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        # Display the uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    with col2:
        if st.button('Classify'):
            # Use the BytesIO object directly in prediction
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
