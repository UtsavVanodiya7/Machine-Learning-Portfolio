import os
import pickle
from argparse import ArgumentParser

from sklearn.metrics import accuracy_score

from text_classification import load_data, preprocess

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-i', '--input_file', required=True, help="Input file path.")
    parser.add_argument('-m', '--model_dir', required=True, help="Directory path of stored model.")

    args = parser.parse_args()
    documents, labels = load_data(args.input_file)
    prefix = args.model_dir + os.sep + args.model_dir

    with open(prefix + '.le', 'rb') as file:
        label_encoder = pickle.load(file)

    with open(prefix + '.clf', 'rb') as file:
        model = pickle.load(file)

    with open(prefix + '.vec', 'rb') as file:
        vectorizer = pickle.load(file)

    new_documents = preprocess(documents)
    test_labels = label_encoder.transform(labels)
    acc_score = accuracy_score(test_labels, model.predict(vectorizer.transform(new_documents)))
    print(f"Accuracy : {acc_score}")
