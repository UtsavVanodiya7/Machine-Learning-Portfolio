import datetime
import os
import pickle
import traceback

import pandas as pd

from argparse import ArgumentParser

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def load_data(input_file):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exists.")
        return

    input_file_df = pd.read_csv(input_file)

    documents = []
    labels = []

    for idx, row in input_file_df.iterrows():
        file_path = row['File Path']
        label = row['Label']

        try:
            with open(file_path, 'r', encoding='utf8') as file:
                text = file.read()

            documents.append(text)
            labels.append(label)
        except:
            print(f"Error while reading file: {file_path}")
            traceback.print_exc()

    return documents, labels


def preprocess(documents, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Encode Labels
    label_encoder = LabelEncoder()
    new_labels = label_encoder.fit_transform(labels)

    with open(output_dir + os.sep + output_dir + '.le', 'wb') as file:
        pickle.dump(label_encoder, file)

    # Remove noise from the documents
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    new_documents = []

    for doc in documents:
        words = [lemmatizer.lemmatize(word) for word in doc.split() if word not in stop_words]
        new_documents.append(" ".join(words))

    return new_documents, new_labels


def train_model(documents, labels, output_dir):
    print("Splitting data in train test set.")
    train_data, test_data, train_labels, test_labels = train_test_split(documents, labels, test_size=0.20,
                                                                        random_state=10)
    print("Building TF-IDF matrix.")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vector = tfidf_vectorizer.fit_transform(train_data)

    with open(output_dir + os.sep + output_dir + '.vec', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)

    # Let's build model.
    print('Training multi-class classifier using Linear SVC.')
    start_time = datetime.datetime.now()
    print(f"Start time : {start_time}")
    model = svm.LinearSVC()
    model.fit(tfidf_vector, train_labels)

    # save the model to disk
    pickle.dump(model, open(output_dir + os.sep + output_dir + '.clf', 'wb'))

    end_time = datetime.datetime.now()

    print(f"Start time : {start_time}")
    print(f"End time : {end_time}\n")

    print(f"Total time : {(end_time - start_time).seconds}")

    print("\nTesting Model.")

    test_tfidf_vector = tfidf_vectorizer.transform(test_data)
    score = model.score(test_tfidf_vector, test_labels)
    print(f"Accuracy : {score}")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-i', '--input_file', required=True, help="Input file path.")
    parser.add_argument('-o', '--output_dir', required=True, help="Directory path to store model.")

    args = parser.parse_args()

    documents, labels = load_data(args.input_file)
    new_documents, new_labels = preprocess(documents, labels, args.output_dir)
    train_model(new_documents, new_labels, args.output_dir)
