import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the dataset
df = pd.read_csv('/content/large.csv')

#  (1: Positive, 0: Neutral, -1: Negative)
X = df['Feedback']
y = df['Sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Display other evaluation metrics

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Store feedback predictions for percentage calculation
feedback_counts = {1: 0, 0: 0, -1: 0}
total_feedbacks = 0


thresholds = {"positive": 60, "neutral": 30, "negative": 10}

def plot_distribution():
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [feedback_counts[1], feedback_counts[0], feedback_counts[-1]]
    colors = ['green', 'blue', 'red']
    plt.figure(figsize=(3, 3))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title("Sentiment Distribution")
    plt.show()

# Manual Testing
print("\nManual Testing:")
while True:
    # Get user input
    user_input = input("Enter student feedback (or 'exit' to stop):")

    # Check for exit condition
    if user_input.lower() == 'exit':
        break

    # Vectorize the user input
    user_input_tfidf = vectorizer.transform([user_input])

    # Make prediction
    prediction = classifier.predict(user_input_tfidf)[0]

    # Update feedback counts
    feedback_counts[prediction] += 1
    total_feedbacks += 1

    # Display the prediction
    if prediction == 1:
        print("Prediction: Positive Feedback\n")
    elif prediction == 0:
        print("Prediction: Neutral Feedback\n")
    else:
        print("Prediction: Negative Feedback\n")

    # Calculate and display sentiment percentages
    pos_percent = (feedback_counts[1] / total_feedbacks) * 100
    neu_percent = (feedback_counts[0] / total_feedbacks) * 100
    neg_percent = (feedback_counts[-1] / total_feedbacks) * 100

    print(f"Sentiment Distribution:")
    print(f"Positive: {pos_percent:.2f}%")
    print(f"Neutral: {neu_percent:.2f}%")
    print(f"Negative: {neg_percent:.2f}%\n")

    # Check if sentiment distribution exceeds thresholds
    if pos_percent >= thresholds["positive"]:
        print(" Majority of feedback is Positive!  YOU CAN UPGRADE YOUR PACE IF YOU WANT")
    elif neg_percent >= thresholds["negative"]:
        print(" Warning: High Negative Feedback!WE HAVE REDUCED YOUR PACE A BIT")
    elif neu_percent >= thresholds["neutral"]:
        print(" Neutral feedback is dominant. YOU ARE IN A GOOD PACE")

    # Plot sentiment distribution
    plot_distribution()
