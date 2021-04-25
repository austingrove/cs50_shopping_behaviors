import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        csvreader.__next__()
        for row in csvreader:
            single_evidence = row[0:9]
            month_map = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8,
                         'Oct': 9, 'Nov': 10, 'Dec': 11}
            single_evidence.append(month_map[row[10]])
            single_evidence.extend(row[11:14])
            single_evidence.append(1 if row[15] == "Returning_Visitor" else 0)
            single_evidence.append(1 if row[16] == "TRUE" else 0)
            evidence.append(single_evidence)
            labels.append(1 if row[-1] == "TRUE" else 0)

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.

    use a KNeighborsClassifier in this function
    """
    neighbors = KNeighborsClassifier(n_neighbors=1)
    neighbors.fit(evidence, labels)
    return neighbors


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    specificity = 0
    for idx, value in enumerate(labels):
        if labels[idx] == predictions[idx]:
            if value == 0:
                specificity += 1
            else:
                sensitivity += 1

    sensitivity /= labels.count(1)
    specificity /= labels.count(0)

    return sensitivity, specificity


if __name__ == "__main__":
    main()

