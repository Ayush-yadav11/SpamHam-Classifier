# Spam/Ham Text Classifier

A Python-based text classification system that uses Natural Language Processing (NLP) and Machine Learning to classify text messages as either spam or legitimate (ham). The system achieves high accuracy in distinguishing between unwanted spam messages and legitimate communications using advanced text preprocessing and the Multinomial Naive Bayes algorithm.

## Features

- Advanced text preprocessing pipeline including:
  - Lowercase conversion
  - Special character and digit removal
  - Tokenization
  - Stopword removal
- Machine learning model using Multinomial Naive Bayes classifier
- Bag-of-words representation using CountVectorizer
- Comprehensive evaluation metrics
- Easy-to-use interface for making predictions
- Sample SMS dataset included

## Requirements

- Python 3.7 or higher
- Required packages:
  - scikit-learn: For machine learning algorithms
  - nltk: For text preprocessing
  - pandas: For data manipulation
  - numpy: For numerical operations

## Installation

1. Clone or download this repository
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. The system will automatically download required NLTK data on first run

## Usage

Run the main script to see the classifier in action:

```bash
python main.py
```

The script will:

1. Load the SMS Spam Collection dataset
2. Split data into training (80%) and testing (20%) sets
3. Train the classifier using preprocessed text data
4. Display classification metrics including accuracy, precision, recall, and F1-score
5. Demonstrate predictions on example messages

## Implementation Details

### Text Preprocessing

- Converts all text to lowercase for consistency
- Removes special characters and digits using regex
- Splits text into individual words (tokenization)
- Removes common English stopwords to reduce noise

### Feature Extraction

- Uses CountVectorizer to convert text into bag-of-words representation
- Creates a sparse matrix of token counts

### Classification

- Implements Multinomial Naive Bayes classifier
- Suitable for discrete features (word counts)
- Performs well with text classification tasks

### Model Evaluation

- Uses standard metrics:
  - Accuracy: Overall correct predictions
  - Precision: Accuracy of spam predictions
  - Recall: Ability to detect all spam messages
  - F1-score: Balanced measure of precision and recall

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
