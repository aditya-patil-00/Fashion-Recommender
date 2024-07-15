import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2
from features import *
from similarity import *
from shopping_api import *

# Define paths to your scripts
FEATURES_SCRIPT_PATH = 'features.py'
SIMILARITY_SCRIPT_PATH = 'similarity.py'
SHOPPING_API_SCRIPT_PATH = 'shopping_api.py'

# Function to resize image to 60x80
def resize_image(image_path):
    image = Image.open(image_path)
    image = image.resize((60, 80))
    return image

def display_images(top5_image_ids, dataset_path, width=100):
    st.write("Top 5 Similar Images:")
    for image_id in top5_image_ids:
        image_id = image_id + '.jpg'
        image_path = os.path.join(dataset_path, image_id)  # Path to your images
        image = mpimg.imread(image_path)
        st.image(image, caption=image_id, width=width)

cwd = os.getcwd()
DATASET_PATH = os.path.join(cwd, '..', 'data', 'images')

# Streamlit app
def main():
    st.title("Fashion Recommender System")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Resize the input image
        image = resize_image(uploaded_file)
        st.image(image, caption='Uploaded Image', width=100)
        st.write("")

        # Save the resized image temporarily
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        model_name = 'resnet50'
        n_components = 100  # Number of PCA components
        pca_model_path = 'pca_model.joblib'

        # Extract features
        features = extract_save(temp_image_path, model_name, n_components, pca_model_path=pca_model_path)
        
        # Find top 5 similar images
        csv_file = 'features.csv'  
        top5_ids = similarity(features, csv_file=csv_file)

        col1, col2 = st.columns(2)

        #display the top 5 images
        with col1:
            display_images(top5_ids, DATASET_PATH)

        # Fetch shopping results
        with col2:
            results = query(top5_ids)
            print(results)
            # Display results
            st.write("Recommendations:")
            for result in results:
                st.write(f"Title: {result[0]}")
                st.write(f"Link: [Click here]({result[1]})")

if __name__ == '__main__':
    main()
