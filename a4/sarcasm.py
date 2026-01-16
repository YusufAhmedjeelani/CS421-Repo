# CS421: Natural Language Processing
# University of Illinois Chicago
# Fall 2025
# Assignment 4
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================


import pandas as pd
import numpy as np
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report

from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, hstack

nltk.download("vader_lexicon", quiet=True)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Function: extract_baseline_features(data, vectorizer, sentiment_intensity)
# data: An array of text instances, with one row per instance
# vectorizer: An instantiated TfidfVectorizer
# sentiment_intensity: An instantiated SentimentIntensityAnalyzer
# Returns: An array of feature vectors, with one row per instance
#
# This function extracts features for the input data, and returns the feature vector.
# Feel free to create similar types of functions for your model!
def extract_baseline_features(data, vectorizer, sentiment_intensity):
    vectorized = vectorizer.transform(data.values.astype('U')) # Get TFIDF vectors
    sentiments = {"neg": [], "neu": [], "pos": [], "compound": [], "num_excl": []}

    i = 0
    for row in data:
        # Add sentiment scores
        scores = sentiment_intensity.polarity_scores(row)
        sentiments["neg"].append(scores["neg"])
        sentiments["neu"].append(scores["neu"])
        sentiments["pos"].append(scores["pos"])
        sentiments["compound"].append(scores["compound"])

        # Count number of exclamation marks
        sentiments["num_excl"].append(row.count("!"))
        i+=1

    df = pd.DataFrame(sentiments)
    features = np.hstack((vectorized.toarray(), df[["neg", "neu", "pos", "compound", "num_excl"]]))
    return features

# Function: baseline(features, labels)
# X_train: A vector of training instances
# y_train: A vector of training labels
# X_test: A vector of test instances
# Returns: Predictions for X_test
#
# This function trains a baseline model and then makes predictions for provided test instances.
def baseline(X_train, y_train, X_test):
    tfidf = TfidfVectorizer().fit(X_train.values.astype('U'))
    sentiment = SentimentIntensityAnalyzer()
    features = extract_baseline_features(X_train, tfidf, sentiment)

    model = SelectKBest(mutual_info_classif, k=500)
    model = GaussianNB()
    model.fit(features, y_train)

    features = extract_baseline_features(X_test, tfidf, sentiment)
    predictions = model.predict(features)

    model_data = {
        "tfidf": tfidf,
        "sentiment": sentiment,
        'model': model
    }

    return predictions, model_data


# Function: sarcasm_model(train, test)
# X_train: A vector of training instances
# y_train: A vector of training labels
# X_test: A vector of test instances
# Returns: Predictions for X_test
#
# This function trains a more advanced model using the training data and labels.
def sarcasm_model(X_train, y_train, X_test):
    random.seed(42)
    
    preds = [random.randint(0, 1) for i in range(0, len(X_test))] # Initialize to just predict random labels
    model = None
    # [WRITE YOUR CODE HERE!  Feel free to create helper functions.]
    #Preprocess and clean data
    cleanXTrain = cleanData(X_train)
    cleanXTest  = cleanData(X_test)
    cleanYTrain = np.asarray(y_train).astype(int)

    wordVec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    charVec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    wordVec.fit(cleanXTrain)
    charVec.fit(cleanXTrain)

    transTrainW = wordVec.transform(cleanXTrain)
    transTrainC = charVec.transform(cleanXTrain)
    TrainFeatures = hstack([transTrainW, transTrainC])

    classifier = LinearSVC(class_weight="balanced", C=1.0, random_state=42)
    classifier.fit(TrainFeatures, cleanYTrain)

    transTestW = wordVec.transform(cleanXTest)
    transTestC = charVec.transform(cleanXTest)
    TestFeatures = hstack([transTestW, transTestC])

    preds = classifier.predict(TestFeatures)
    preds = [int(p) for p in preds]

    model = {
        "wordVec": wordVec,
        "charVec": charVec,
        "model": classifier 
    }

    return preds, model # Return your predictions, similarly to the baseline

def cleanData(x):
    if isinstance(x, pd.Series):
        return x.astype(str).tolist()

    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0].astype(str).tolist()

    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [str(v) for v in x]

    return [str(x)]


# Function: get_sarcasm_predictions_en(test_set_en, model)
# test_set_en: An input test set
# model: A trained model
# Returns: A list of predictions
#
# This function  returns the list of predictions obtained from the en sarcasm model for test_set_en.
def get_sarcasm_predictions_en(test_set_en, model):
    predictions_en = [random.randint(0, 1) for i in range(0, len(test_set_en))] # Initialize to just predict random labels
    
    # [WRITE YOUR CODE HERE!  Feel free to create helper functions.]

    cleanText = cleanData(test_set_en)
    transTestW = model["wordVec"].transform(cleanText)
    transTestC = model["charVec"].transform(cleanText)
    TestFeatures = hstack([transTestW, transTestC])
    predictions_en = model["model"].predict(TestFeatures)
    predictions_en = [int(p) for p in predictions_en]

    with open("en_predictions.txt", "w") as f:
        for prediction in predictions_en:
            f.write(str(prediction) + "\n")
  
    print("done")

    return predictions_en # Return your predictions, similarly to the baseline



# Function: arabic_sarcasm_model(X_train, y_train, X_test):
# X_train: A vector of training instances
# y_train: A vector of training labels
# X_test: A vector of test instances
# Returns: Predictions for X_test
#
# This function trains a more advanced model using the provided feature vectors and labels.  Since it
# is designed for use with Arabic data, it may or may not vary from sarcasm_model().
def arabic_sarcasm_model(X_train, y_train, X_test):
    random.seed(42)
    predictions = [random.randint(0, 1) for i in range(0, len(X_test))]  # Initialize to just predict random labels
    model = None
    
    # [WRITE YOUR CODE HERE!  Feel free to create helper functions.]

    return predictions, model # Return your predictions, similarly to the baseline


# Function: get_sarcasm_predictions_ar(test_set_ar, model)
# input: An input test set
# model: A trained model
# Returns: A list of predictions
#
# This function  returns the list of predictions obtained from the en sarcasm model for test_set_ar.
def get_sarcasm_predictions_ar(test_set_ar, model):
    predictions_ar = [random.randint(0, 1) for i in
                      range(0, len(test_set_ar))]  # Initialize to just predict random labels

    # [WRITE YOUR CODE HERE!  Feel free to create helper functions.]

    return predictions_ar  # Return your predictions, similarly to the baseline


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python sarcasm.py
# It should produce the following output (with correct solution); the performance of your model will
# be shown in the "Updated Performance" sections:
#
# $ python3 sarcasm.py
# Baseline Performance:
#               precision    recall  f1-score   support
#
#            0       0.77      0.61      0.68       524
#            1       0.23      0.38      0.28       156
#
#     accuracy                           0.56       680
#    macro avg       0.50      0.50      0.48       680
# weighted avg       0.64      0.56      0.59       680
#
# Updated Performance:
#               precision    recall  f1-score   support
#
#            0       0.75      0.49      0.59       524
#            1       0.21      0.44      0.28       156
#
#     accuracy                           0.48       680
#    macro avg       0.48      0.47      0.44       680
# weighted avg       0.62      0.48      0.52       680
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Arabic Baseline Performance:
#               precision    recall  f1-score   support
#
#            0       0.87      0.93      0.90       487
#            1       0.66      0.51      0.58       133
#
#     accuracy                           0.84       620
#    macro avg       0.77      0.72      0.74       620
# weighted avg       0.83      0.84      0.83       620
#
# Arabic Updated Performance:
#               precision    recall  f1-score   support
#
#            0       0.76      0.49      0.60       487
#            1       0.19      0.44      0.27       133
#
#     accuracy                           0.48       620
#    macro avg       0.48      0.47      0.43       620
# weighted avg       0.64      0.48      0.53       620
def main():
    df = pd.read_csv("train.En.csv")
    # print(df.to_string()) # Uncomment if you'd like to see the data you read in

    # Split data into training and development sets
    train = df.sample(frac=0.8, random_state=200)
    dev = df.drop(train.index)

    X_train = train["text"]
    y_train = train["sarcastic"]

    X_dev = dev["text"]
    y_dev = dev["sarcastic"]

    # reading the english test set
    test_set_en = pd.read_csv("test_en.csv")
    # Train baseline model, get predictions, and analyze output
    predictions, baseline_model_en = baseline(X_train, y_train, X_dev)
    print("Baseline Performance:\n" + classification_report(y_dev, predictions))

    # Train the updated model, get predictions, and analyze output
    predictions, new_model = sarcasm_model(X_train, y_train, X_dev)
    print("Updated Performance:\n" + classification_report(y_dev, predictions))


    # Get predictions for test_set_en and analyze the output
    predictions_en = get_sarcasm_predictions_en(test_set_en, new_model)
    print(predictions_en)
    print("Test set predictions (English): \n" + classification_report(test_set_en["sarcastic"], predictions_en))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""    # Bonus points (comment out if you are not interested in this part)!
    df_ar = pd.read_csv("train.Ar.csv")

    # Split data into training and development sets
    train_ar = df_ar.sample(frac=0.8, random_state=200)
    dev_ar = df_ar.drop(train_ar.index)

    X_train_ar = train_ar["text"]
    y_train_ar = train_ar["sarcastic"]

    X_dev_ar = dev_ar["text"]
    y_dev_ar = dev_ar["sarcastic"]

    # reading the arabic test set
    test_set_ar = pd.read_csv("test_ar.csv")

    # Train baseline model, get predictions, and analyze output
    predictions, baseline_model_ar = baseline(X_train_ar, y_train_ar, X_dev_ar)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
          "Arabic Baseline Performance:\n" + classification_report(y_dev_ar, predictions))

    # Train the updated model, get predictions, and analyze output
    predictions_ar, new_model_ar = arabic_sarcasm_model(X_train_ar, y_train_ar, X_dev_ar)
    print("Arabic Updated Performance:\n" + classification_report(y_dev_ar, predictions_ar))


    # Get predictions for test_set_en and analyze the output
    predictions_ar = get_sarcasm_predictions_ar(test_set_ar, new_model_ar)
    print(predictions_ar)
    print("Test set predictions (Arabic) updated model: \n" + classification_report(test_set_ar["sarcastic"], predictions_ar))
"""
    


################ Do not make any changes below this line ################
if __name__ == '__main__':
	exit(main())

