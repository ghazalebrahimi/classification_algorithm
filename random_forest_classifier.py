from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier(train_set, valid_set, test_set):

    clf = RandomForestClassifier()
    clf.fit(train_set[0], train_set[1])

    valid_predict = clf.predict(valid_set[0])
    wrong_answer = 0
    for i, j in zip(valid_predict, valid_set[1]):
        if i != j:
            wrong_answer += 1
    print("Random Forest Classifier predicts the validation set with accuracy of: ",
          100 * (len(valid_predict) - wrong_answer) / (len(valid_predict)))

    test_predict = clf.predict(test_set[0])
    wrong_answer = 0
    for i, j in zip(test_predict, test_set[1]):
        if i != j:
            wrong_answer += 1
    print("Random Forest Classifier predicts the test set with accuracy of: ",
          100 * (len(test_predict) - wrong_answer) / (len(test_predict)))
