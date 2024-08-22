import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# Load the IMDb movie review dataset
data = pd.read_csv('imdb_reviews.csv')

# Preprocessing the data
data['review'] = data['review'].apply(lambda x: re.sub('<.*?>', '', x))
data['review'] = data['review'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
data['review'] = data['review'].apply(lambda x: x.lower())
stop_words = set(stopwords.words('english'))
data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
lemmatizer = WordNetLemmatizer()# Lemmatize the text data
data['review'] = data['review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Convert the text into a numerical format
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['review'])
y = data['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X, y, cv=5)
model.fit(X_train, y_train)

# Save the model and vectorizer using pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Predict probabilities for the test set
probabilities = model.predict_proba(X_test)

# The probabilities for the positive class (1)
positive_probabilities = probabilities[:, 1]

# Round the probabilities to 10 decimal places
positive_probabilities = np.around(positive_probabilities, decimals=10)

# Example output
print("First 10 probability predictions for the positive class:")
print(positive_probabilities[:10])
