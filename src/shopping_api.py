import os
from serpapi import GoogleSearch
import pandas as pd
from dotenv import load_dotenv
import csv

load_dotenv()
API_KEY = os.getenv("SERPAPI_API_KEY")  

def search_google_shopping(query):
    params = {
  "api_key": API_KEY,
  "engine": "google",
  "q": query,
  "location": "India",
  "google_domain": "google.co.in",
  "gl": "in",
  "hl": "en"
}
    search = GoogleSearch(params)
    results = search.get_dict()

    tup = ()
    for result in results['organic_results']:
        tup = (result['title'], result['link'])

    return tup

# Example usage
def query(img_ids):
    curr_dir = os.getcwd()
    DATASET_PATH = os.path.join(curr_dir, 'data')
    # Load the styles.csv dataset
    styles_df = pd.read_csv(os.path.join(DATASET_PATH, 'styles.csv')) 

    # Ensure 'id' column is of string type
    styles_df['id'] = styles_df['id'].astype(str)
    img_ids = [str(img_id) for img_id in img_ids]

    # Debug: Print IDs and DataFrame
    #print(f"Image IDs: {img_ids}")
    #print(styles_df.head())

    titles = []
    for img_id in img_ids:
        matching_rows = styles_df.loc[styles_df['id'] == img_id, 'productDisplayName']
        if not matching_rows.empty:
            title = matching_rows.iloc[0]
            titles.append(title)
        else:
            print(f"No match found for img_id: {img_id}")

    shopping_results = []
    for title in titles:
        print(f"Searching for: {title}")
        shopping_results.append(search_google_shopping(title))

    return shopping_results
