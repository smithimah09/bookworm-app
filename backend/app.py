import json
import numpy as np
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the book dataset
def load_books(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Tokenize text into words
def tokenize(text):
    return text.lower().split()

# Create a term frequency vector for a given text
def term_frequency(text, vocab):
    words = tokenize(text)
    word_count = Counter(words)
    return np.array([word_count[word] for word in vocab])

# Calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# Get recommendations based on the queried book title
def get_book_recommendations(books, queried_title, num_recommendations=5):
    titles = [book['title'] for book in books]
    summaries = [book['summary'] for book in books]

    # Create a vocabulary from all summaries
    vocab = set(tokenize(' '.join(summaries)))

    # Create term frequency vectors for each summary
    tf_vectors = np.array([term_frequency(summary, vocab) for summary in summaries])

    # Find the index of the queried book title
    if queried_title not in titles:
        return f"Book titled '{queried_title}' not found in the dataset."
    
    queried_index = titles.index(queried_title)

    # Calculate cosine similarity between the queried book and all other books
    similarities = np.array([cosine_similarity(tf_vectors[queried_index], tf_vectors[i]) for i in range(len(tf_vectors))])

    # Get indices of the most similar books (excluding the queried book itself)
    similar_indices = np.argsort(similarities)[::-1][1:num_recommendations + 1]

    # Return the titles of the recommended books
    recommended_titles = [titles[i] for i in similar_indices]
    return recommended_titles

# Create a Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
books = load_books('books.json')  # Load the book dataset

# Define a route for the recommendation API
@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Please provide a book title."}), 400

    recommendations = get_book_recommendations(books, title, num_recommendations=5)
    return jsonify({"recommendations": recommendations})

# Define a route for getting book information
@app.route('/book_info', methods=['GET'])
def book_info():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Please provide a book title."}), 400
    
    # Search for the book in the dataset
    for book in books:
        if book['title'].lower() == title.lower():
            return jsonify(book)  # Return the book details as JSON

    return jsonify({"error": f"Book titled '{title}' not found in the dataset."}), 404

# Run the app
if __name__ == "__main__":
    app.run(port=5000)