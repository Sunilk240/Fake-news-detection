import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set page background
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TfidfVectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to preprocess input text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])
    return text

# Streamlit app
def main():
    st.title("Fake News Detection")

    # Text input for user to enter news content
    news_content = st.text_area("Enter the news content")

    if st.button("Check"):
        # Preprocess the input text
        preprocessed_text = preprocess_text(news_content)
        # Vectorize the preprocessed text
        vectorized_text = vectorizer.transform([preprocessed_text])
        # Make prediction
        prediction = model.predict(vectorized_text)

        if prediction[0] == 1:
            st.error("The news is classified as REAL.")
        else:
            st.error("The news is classified as FAKE.")

    

if __name__ == '__main__':
    main()
