import weaviate
from weaviate.classes.init import Auth
import weaviate.classes.config as wc

import pandas as pd
import requests
from datetime import datetime, timezone
import json
from weaviate.util import generate_uuid5
from tqdm import tqdm

import weaviate.classes.query as wq
import os


headers = {
    "X-OpenAI-Api-Key": "OPENAI_API_KEY"
}  # Replace with your OpenAI API key

client = weaviate.connect_to_weaviate_cloud(
    cluster_url="WCD_URL",  # Replace with your WCD URL
    auth_credentials=Auth.api_key(
        "WCD_KEY"
    ),  # Replace with your WCD key
    headers=headers,
)
'''
print(client.is_ready())
client.collections.create(
    name="Books",
    properties=[
        wc.Property(name="title", data_type=wc.DataType.TEXT),
        wc.Property(name="authors", data_type=wc.DataType.TEXT),
        wc.Property(name="average_rating", data_type=wc.DataType.NUMBER),
        wc.Property(name="genre", data_type=wc.DataType.TEXT),
        wc.Property(name="cover", data_type=wc.DataType.TEXT), # need to get image from that url
        wc.Property(name="summary", data_type=wc.DataType.TEXT),
    ],
    # Define the vectorizer module
    vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(),
    # Define the generative module
    generative_config=wc.Configure.Generative.openai()
)
'''
data_url = "https://raw.githubusercontent.com/shahsalonik/bookworm/refs/heads/main/data/bookworm_dataset.json"
resp = requests.get(data_url)
df = pd.DataFrame(resp.json())

# Get the collection
books = client.collections.get("Books")

# Enter context manager
with books.batch.dynamic() as batch:
    # Loop through the data
    for i, book in tqdm(df.iterrows()):

        # Build the object payload
        book_obj = {
            "title": book["title"],
            "authors": book["authors"],
            "average_rating": book["average_rating"],
            "genre": book["genre"],
            "cover": book["cover"],
            "summary": book["summary"],
        }

        # Add object to batch queue
        batch.add_object(
            properties=book_obj,
            # references=reference_obj  # You can add references here
        )
        # Batcher automatically sends batches

# Check for failed objects
if len(books.batch.failed_objects) > 0:
    print(f"Failed to import {len(books.batch.failed_objects)} objects")

books = client.collections.get("Books")

# Perform query
response = books.generate.near_text(
    query="Harry Potter and the Goblet of Fire",
    limit=1,
    single_prompt="Generate a list of books most similar to this input excluding the input itself: {summary}"
)

# Inspect the response
for o in response.objects:
    print(o.properties["title"])  # Print the title
    print(o.generated)  # Print the generated text (the title, in French)

client.close()