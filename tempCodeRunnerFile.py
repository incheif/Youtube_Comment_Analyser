import pickle
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify, render_template
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build

# Load the previously saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def get_youtube_comments(video_id, api_key, max_results=100):
    youtube = build('youtube', 'v3', developerKey=api_key)

    comments = []
    next_page_token = None

    while len(comments) < max_results:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=min(100, max_results - len(comments)),
            textFormat="plainText"
        ).execute()

        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            if len(comment.split()) <= 20:
                comments.append(comment)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments

def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith(('/embed/', '/v/')):
            return parsed_url.path.split('/')[2]
    return None

def preprocess_comments(new_data):
    new_data['Comment'] = new_data['Comment'].apply(lambda x: re.sub('<.*?>', '', x))
    new_data['Comment'] = new_data['Comment'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
    new_data['Comment'] = new_data['Comment'].apply(lambda x: x.lower())
    stop_words = set(stopwords.words('english'))
    new_data['Comment'] = new_data['Comment'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    new_data['Comment'] = new_data['Comment'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    return new_data

   

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    youtube_url = request.json['url']
    video_id = get_video_id(youtube_url)

    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    api_key = "AIzaSyA5lZYI3wMZh3NigNipLtA3ZXuuHo7uWjU"  # Replace with your actual API key
    comments = get_youtube_comments(video_id, api_key, max_results=500)

    df = pd.DataFrame(comments, columns=['Comment'])
    processed_df = preprocess_comments(df)
    X_new = vectorizer.transform(processed_df['Comment'])

    # Predict probabilities for the new data
    probabilities = model.predict_proba(X_new)

    # Calculate average sentiment
    avg_sentiment = np.mean(probabilities[:, 1])

    # Count positive and negative comments
    positive_count = np.sum(probabilities[:, 1] > 0.5)
    negative_count = len(probabilities) - positive_count

    results = {
        'average_sentiment': float(avg_sentiment),
        'positive_count': int(positive_count),
        'negative_count': int(negative_count),
        'total_comments': len(comments)
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)