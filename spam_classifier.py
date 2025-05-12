# Import required libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For dataset splitting
from sklearn.feature_extraction.text import CountVectorizer  # For text vectorization
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation
import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords  # For removing common words
import re  # For regular expressions
import string  # For string operations

class SpamHamClassifier:
    """
    A text classifier that distinguishes between spam and ham (non-spam) messages.
    Uses CountVectorizer for text vectorization and Multinomial Naive Bayes for classification.
    Includes text preprocessing steps like lowercase conversion, special character removal,
    and stopword removal.
    """
    def __init__(self):
        # Initialize the vectorizer and classifier
        self.vectorizer = CountVectorizer()  # Converts text to bag-of-words representation
        self.classifier = MultinomialNB()  # Naive Bayes classifier for discrete features
        
    def preprocess_text(self, text):
        """
        Preprocess the input text by performing several cleaning steps.
        
        Args:
            text (str): The input text to be preprocessed
            
        Returns:
            str: The preprocessed text
        """
        # Convert to lowercase for consistency
        text = text.lower()
        # Remove special characters and digits using regex
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Split text into individual words
        tokens = text.split()
        # Remove common English stopwords to reduce noise
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Join tokens back into a single string
        return ' '.join(tokens)
    
    def train(self, X_train, y_train):
        """
        Train the classifier on the preprocessed text data.
        
        Args:
            X_train (list): List of training text messages
            y_train (list): List of corresponding labels (1 for spam, 0 for ham)
        """
        # Apply preprocessing to all training texts
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        # Convert text to numerical features using bag-of-words
        X_train_vectorized = self.vectorizer.fit_transform(X_train_processed)
        # Train the Naive Bayes classifier
        self.classifier.fit(X_train_vectorized, y_train)
    
    def predict(self, X_test):
        """
        Predict spam/ham labels for new messages.
        
        Args:
            X_test (list): List of text messages to classify
            
        Returns:
            array: Predicted labels (1 for spam, 0 for ham)
        """
        # Preprocess test data using same steps as training
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        # Transform text using fitted vectorizer
        X_test_vectorized = self.vectorizer.transform(X_test_processed)
        # Make predictions using trained classifier
        return self.classifier.predict(X_test_vectorized)

# Download required NLTK data for text processing
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords removal