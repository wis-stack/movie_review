import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from joblib import load
import os

# Download NLTK resources if not already downloaded
nltk.download('punkt_tab')

# Load the saved Naive Bayes classifier
bayes_classifier = load('bayes_classifier.joblib')

# Function to extract features from a movie review
def review_features(review, top_keys):
    review_words = set(word_tokenize(review))
    features = {}
    for word in top_keys:
        features[word] = (word in review_words)
    return features

# Load top keys (most informative features used during training)
def load_top_keys():
    with open('top_keys.txt', 'r') as f:
        top_keys = [word.strip() for word in f.readlines()]
    return top_keys

# Streamlit app
def main():
    st.title('Movie Review Sentiment Analysis')
    st.write('Enter a movie review or upload a text file to predict its sentiment.')

    # File upload
    uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
    if uploaded_file is not None:
        # Read file contents
        raw_text = uploaded_file.read().decode('utf-8')
        st.text_area("File content", raw_text, height=200)
        review = raw_text
    else:
        review = st.text_area('Or enter a movie review:', height=200)

    # Load top keys
    top_keys = load_top_keys()

    if st.button('Predict Sentiment'):
        if review.strip() != '':
            # Classify the review using the loaded classifier
            features = review_features(review, top_keys)
            sentiment = bayes_classifier.classify(features)
            
            # Display prediction
            st.success(f'The predicted sentiment for the review is: {sentiment}')
        else:
            st.warning('Please enter a movie review or upload a text file.')

if __name__ == '__main__':
    main()
