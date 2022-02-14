from textblob.classifiers import NaiveBayesClassifier

import sklearn.metrics as sk

train = []

test = []
with open('Software_neg1_to_1_20.csv', 'r') as csvfile:
    for line in csvfile:
        tup = line.split(",", maxsplit=1)
        try:
            if float(tup[0]) > 0:
                review = (tup[1], "pos")
            elif float(tup[0]) == 0:
                continue
            else:
                review = (tup[1], "neg")

            train.append(review)

            print(tup)
        except ValueError:
            continue

print("training 200")
with open('Software_neg1_to_1_200.csv', 'r') as csvfile:
    for line in csvfile:
        tup = line.split(",", maxsplit=1)
        try:
            if float(tup[0]) > 0:
                review = (tup[1], "pos")
            elif float(tup[0]) == 0:
                continue
            else:
                review = (tup[1], "neg")

            train.append(review)
        except ValueError:
            continue

cl = NaiveBayesClassifier(train)

print("training 1000")
y_true = []
y_pred = []
with open('Software_neg1_to_1_1000.csv', 'r') as csvfile:
    for line in csvfile:
        tup = line.split(",", maxsplit=1)
        try:
            if float(tup[0]) > 0:
                y_true.append(1)
            elif float(tup[0]) == 0:
                continue
            else:
                y_true.append(0)

            if cl.classify(tup[1]) == 'pos':
                y_pred.append(1)
            else:
                y_pred.append(0)
        except ValueError:
            continue

print("Precision: " + str(sk.precision_score(y_true, y_pred)))
print("recall: " + str(sk.recall_score(y_true, y_pred)))
print("f1-score: " + str(sk.f1_score(y_true, y_pred)))
print("accuracy: " + str(sk.accuracy_score(y_true, y_pred)))
