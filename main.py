# Import required libraries and custom classifier
from spam_classifier import SpamHamClassifier  # Custom classifier implementation
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation

def load_sms_data():
    """
    Load and preprocess the SMS Spam Collection dataset.
    
    Returns:
        tuple: (messages, labels) where messages is a list of SMS texts and 
               labels is a list of binary values (1 for spam, 0 for ham)
    """
    # Initialize empty lists to store messages and their labels
    messages = []
    labels = []
    
    with open('SMSSpamCollection', 'r', encoding='utf-8') as file:
        for line in file:
            # Split by tab as per the dataset format
            label, message = line.strip().split('\t')
            messages.append(message)
            # Convert 'spam' to 1 and 'ham' to 0 for binary classification
            labels.append(1 if label.lower() == 'spam' else 0)
    
    return messages, labels

def main():
    """
    Main function to run the spam classification pipeline:
    1. Load and preprocess the SMS dataset
    2. Split data into training and testing sets
    3. Train the classifier
    4. Evaluate performance
    5. Test with new example messages
    """
    # Load SMS spam collection data
    messages, labels = load_sms_data()
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42  # Fixed random state for reproducibility
    )
    
    # Initialize and train the classifier
    classifier = SpamHamClassifier()
    classifier.train(X_train, y_train)
    
    # Make predictions on test set
    predictions = classifier.predict(X_test)
    
    # Print model performance metrics
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.2f}")
    
    # Test the model with example messages
    new_messages = [
        "Hey, what time is the meeting tomorrow?",  # Example of a legitimate message
        "CONGRATULATIONS! You've won a free cruise! Click here!"  # Example of a spam message
    ]
    
    # Get predictions for new messages
    new_predictions = classifier.predict(new_messages)
    print("\nPredictions for new messages:")
    for message, prediction in zip(new_messages, new_predictions):
        result = "Spam" if prediction == 1 else "Ham"
        print(f"Message: {message}\nPrediction: {result}\n")

# Entry point of the script
if __name__ == "__main__":
    main()