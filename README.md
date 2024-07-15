<h1 align="center" id="title">Fashion-Recommender</h1>

<p id="description">Dataset source: <a href="https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small" target="_blank">Fashion Product Images Small on Kaggle</a></p>

<h2>Project Screenshots:</h2>

<img src="https://imgur.com/a/rPgkU9D" alt="project-screenshot" width="400" height="400/">

<img src="https://imgur.com/a/GcLqVPL" alt="project-screenshot" width="400" height="400/">

  
  
<h2> Features</h2>

Here're some of the project's best features:

*   Image Upload and Resizing: Upload an image of a fashion item which is then resized to 60x80 pixels for uniformity in processing.
*   Feature Extraction: Utilizes a pre-trained ResNet50 model to extract features from the uploaded image ensuring accurate similarity comparisons.
*   Dimensionality Reduction: Applies Principal Component Analysis (PCA) to reduce the dimensionality of the extracted features enhancing performance and efficiency.
*   Similarity Matching: Compares the uploaded image's features with a dataset of fashion items to identify the top 5 most similar items using cosine similarity.
*   Shopping Results Fetching: Uses the SERP API to fetch real-time shopping results including titles links and prices for the recommended items.
*   Interactive User Interface: Built with Streamlit providing an intuitive and interactive web interface for users to upload images and view recommendations seamlessly.
*   Data Management: Organizes and manages a dataset of fashion items including images and associated metadata for robust and accurate recommendations.
*   Secure API Key Management: Utilizes Streamlit's secrets management to securely handle API keys ensuring sensitive information is protected.
*   Deployment-Ready: Easily deployable on Streamlit Cloud with relative file paths and configuration settings optimized for cloud hosting.

<h2> Installation Steps:</h2>

<p>1. Clone the project</p>

```
  git clone https://github.com/aditya-patil-00/Fashion-Recommender
```

<p>2. Go to project directory</p>

```
cd Fashion-Recommender
```

<p>3. Install dependencies</p>

```
pip install -r requirements.txt
```

<p>4. Start the server</p>

```
streamlit run src/app.py
```
