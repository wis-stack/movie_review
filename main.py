import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from joblib import load

# Download NLTK resources if not already downloaded
nltk.download('punkt')

# Load the saved Naive Bayes classifier
bayes_classifier = load('bayes_classifier.joblib')

# Function to extract features from a movie review
def review_features(review):
    review_words = set(word_tokenize(review))
    features = {}
    for word in top_keys:
        features[word] = (word in review_words)
    return features

# Load top keys (most informative features used during training)
with open('top_keys.txt', 'r') as f:
    top_keys = [word.strip() for word in f.readlines()]

# Streamlit app
def main():
    st.title('Movie Review Sentiment Analysis')
    st.write('Enter a movie review to predict its sentiment.')

    # Input for movie review
    review = st.text_area('Review:')
    
    if st.button('Predict Sentiment'):
        if review.strip() != '':
            # Classify the review using the loaded classifier
            features = review_features(review)
            sentiment = bayes_classifier.classify(features)
            
            # Display prediction
            st.success(f'The predicted sentiment for the review is: {sentiment}')
        else:
            st.warning('Please enter a movie review.')

if __name__ == '__main__':
    main()
