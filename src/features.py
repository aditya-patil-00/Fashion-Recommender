import torch
import cv2
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import joblib

class FeatureExtractor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.get_model()
        self.model.eval()
        self.features = []
        self.hook = self.register_hook()
        self.transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def get_model(self):
        if self.model_name == 'resnet18':
            return models.resnet18(pretrained=True)
        elif self.model_name == 'resnet50':
            return models.resnet50(pretrained=True)
        elif self.model_name == 'vgg16':
            return models.vgg16(pretrained=True)
        elif self.model_name == 'vgg19':
            return models.vgg19(pretrained=True)
        elif self.model_name == 'alexnet':
            return models.alexnet(pretrained=True)
        else:
            raise ValueError('Invalid model name')
    
    def register_hook(self):
        if self.model_name in ['resnet18', 'resnet50']:
            layer = self.model.avgpool
        elif self.model_name in ['vgg16', 'vgg19', 'alexnet']:
            layer = self.model.classifier[-2]
        else:
            raise ValueError('Invalid model name')

        return layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features.append(output)
        
    def extract_features(self, img):
        self.features = []  # Reset features
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            self.model(img)
        # Assuming avgpool or classifier[-2] layer output shape is (batch_size, feature_dim, 1, 1)
        return self.features[0].squeeze().numpy()

    def close(self):
        self.hook.remove()

def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure image is in RGB format
    return img

def extract(imgs, model_name):
    model = FeatureExtractor(model_name)
    features = []
    
    # Check if imgs is a list (multiple images) or a single image
    if isinstance(imgs, list):
        for img in imgs:
            img = load_image(img)
            extracted_features = model.extract_features(img)
            print(f"Extracted features shape for {img}: {extracted_features.shape}")  # Debug print
            features.append(extracted_features)
    else:
        # Assuming imgs is a single image path or image data
        img = load_image(imgs)
        extracted_features = model.extract_features(img)
        print(f"Extracted features shape for {img}: {extracted_features.shape}")  # Debug print
        features.append(extracted_features)
    
    model.close()
    return np.array(features)

def apply_pca(features, n_components, save_model_path):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    joblib.dump(pca, save_model_path)  # Save the PCA model
    return features_pca

def transform_with_pca(features, pca_model_path):
    pca = joblib.load(pca_model_path)
    return pca.transform(features)

def save_features_to_csv(features_pca, image_ids, save_path):
    df = pd.DataFrame(features_pca)
    df['id'] = image_ids
    df = df[['id'] + [col for col in df.columns if col != 'id']]
    df.to_csv(save_path, index=False)

def extract_save(imgs, model_name, n_components, save_path=None, pca_model_path=None):
    # Extract features from images
    features = extract(imgs, model_name)
    
    print(f"Extracted features shape: {features.shape}")  # Debug print
    
    # Apply PCA and save the model if extracting from multiple images
    if isinstance(imgs, list):
        if n_components > features.shape[1]:
            raise ValueError("n_components must be less than or equal to the number of original features")
        
        features_pca = apply_pca(features, n_components, pca_model_path)
    else:
        features_pca = transform_with_pca(features, pca_model_path)
    
    print(f"Features after PCA: {features_pca.shape}")  # Debug print
    
    # Extract image IDs from file paths
    if isinstance(imgs, list):
        image_ids = [os.path.basename(img) for img in imgs]
    else:
        image_ids = [os.path.basename(imgs)]  # For a single image

    # Save to CSV along with image ids, if save_path is provided
    if save_path:
        save_features_to_csv(features_pca, image_ids, save_path)
    
    # Return features for a single image or DataFrame for multiple images
    if not isinstance(imgs, list):
        return features_pca
    
    return features_pca

# Example usage:
curr_dir = os.getcwd()
DATASET_PATH = os.path.join(curr_dir, '..', 'data', 'images')

def img_path(img):
    return os.path.join(DATASET_PATH, img)

def get_all_images(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Get paths to all images in DATASET_PATH
#imgs = get_all_images(DATASET_PATH)
model_name = 'resnet50'
n_components = 100  # Number of PCA components
save_path = 'features.csv'
pca_model_path = 'pca_model.joblib'

# Extract and save features for all images
#extract_save(imgs, model_name, n_components, save_path)

# Example usage for a single image
#img = 'shirt_image.jpg'
#image_path = img_path(img)
#single_image_features = extract_save(image_path, model_name, n_components, pca_model_path=pca_model_path)
#print("Extracted features for single image:", single_image_features.shape)
#save it in txt file
#np.savetxt('features.txt', single_image_features, delimiter=',')
