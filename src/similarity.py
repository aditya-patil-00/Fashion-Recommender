import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def extract_numeric_id(image_id):
    # Assuming image_id is in the format 'XXXXX.jpg'
    numeric_id = image_id.split('.')[0]  
    return numeric_id

def similarity(input_features, csv_file):
    # Load features from CSV file
    csv_data = np.genfromtxt(csv_file, delimiter=',', dtype=str, skip_header=1)
    csv_features = csv_data[:, 1:]  # Assuming features start from the second column
    image_ids = csv_data[:, 0]  # Assuming first column contains image IDs
    
    # Debug prints
    print(f"Features from CSV: {csv_features.shape}")
    print(f"Image IDs: {image_ids.shape}")
    
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(input_features, csv_features)
    
    # Get top 5 similar images
    top5_indices = similarity_scores.argsort()[0][-5:][::-1]
    top5_image_ids = image_ids[top5_indices]

    # Extract numeric IDs
    top5_numeric_ids = [extract_numeric_id(image_id) for image_id in top5_image_ids]
    
    return top5_numeric_ids

# Example usage
#input_features = np.loadtxt('features.txt', delimiter=',')
#input_features = input_features.reshape(1, -1) 
#csv_file = 'features.csv'  

#top5_image_ids = similarity(input_features, csv_file)
#print("Top 5 Image IDs:", top5_image_ids)

def img_path(img):
    curr_dir = os.getcwd()
    DATASET_PATH = os.path.join(curr_dir, '..', 'data', 'images')
    return os.path.join(DATASET_PATH, img)

def display_images(top5_image_ids):
    # Display top 5 images in one window
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i, image_id in enumerate(top5_image_ids):
        image_id = image_id + '.jpg'  
        image_path = img_path(image_id) # Replace with the actual path to your images
        image = mpimg.imread(image_path)
        axs[i].imshow(image)
        axs[i].axis('off')
    plt.show()

#display_images(top5_image_ids)
