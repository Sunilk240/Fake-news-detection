# Fake-news-detection
Fake news detection using Decision Tree Algorithm 

Certainly! Here are the README files for both the initial code and the Streamlit web application:

## Initial Code README(DT.ipynb)

# Fake News Detector

This Python code detects fake news using machine learning techniques. It performs the following tasks:

1. Data Collection: The code reads a CSV file called 'WELFake_Dataset.csv' that contains news articles and their labels (0 for fake news, 1 for real news).
2. Data Cleaning: It removes missing values, shuffles the data, removes duplicates, and performs text preprocessing tasks such as converting text to lowercase, removing URLs and punctuation, removing stopwords, and lemmatizing words.
3. Exploratory Data Analysis (EDA): It visualizes the distribution of news labels, news lengths, and the frequency of the most common words in the news articles.
4. Data Preprocessing: It separates the data into independent variables (X) and the dependent variable (y). It then converts the textual data into numerical data using TF-IDF vectorization.
5. Model Training: It trains a Decision Tree classifier on the training set and evaluates its performance on both the training and testing sets. It also performs regularization by adjusting the maximum depth and minimum samples split parameters of the Decision Tree classifier.
6. Model Saving: It saves the trained model and TF-IDF vectorizer to files named 'model.pkl' and 'vectorizer.pkl', respectively.
7. Fake News Detection: It provides a Streamlit web interface where users can enter a news article, and the program predicts whether it is real or fake using the trained model.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- re
- nltk
- scikit-learn
- streamlit

## Usage

1. Install the required packages mentioned above.
2. Place the 'WELFake_Dataset.csv' file in the same directory as the code.
3. Run the code using the Python interpreter.
4. After the program finishes running, the trained model ('model.pkl') and TF-IDF vectorizer ('vectorizer.pkl') will be saved in the same directory.
5. To use the fake news detector, run the Streamlit web application by following the instructions in the Streamlit Web Application README.

## Contributing

Feel free to contribute to this project by submitting bug reports, feature requests, or pull requests.


## Streamlit Web Application README(streamlitapp.py)

# Fake News Detection Web Application

This is a Streamlit web application that uses a trained machine learning model to detect fake news. It allows users to enter news content and displays whether the news is classified as real or fake. The app utilizes the trained model and vectorizer from the files 'model.pkl' and 'vectorizer.pkl', respectively.

## Requirements

- Python 3.x
- streamlit
- pickle
- re
- nltk

## Usage

1. Install the required packages mentioned above.
2. Place the 'model.pkl' and 'vectorizer.pkl' files in the same directory as the 'streamlitapp.py' file.
3. Run the following command in the terminal:

```
streamlit run streamlitapp.py
```

4. The web application will start running, and a local URL will be provided (e.g., `http://localhost:8501`).
5. Open a web browser and enter the provided URL.
6. In the text area, enter the news content you want to check.
7. Click on the "Check" button.
8. The app will process the input, make a prediction using the trained model, and display whether the news is classified as real or fake.


