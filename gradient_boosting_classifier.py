from sklearn.ensemble import GradientBoostingClassifier


def gradient_boosting_classifier(train_set, valid_set, test_set):

    clf = GradientBoostingClassifier()
    clf.fit(train_set[0][:1000], train_set[1][:1000])

    valid_predict = clf.predict(valid_set[0])
    wrong_answer = 0
    for i, j in zip(valid_predict, valid_set[1]):
        if i != j:
            wrong_answer += 1
    print("Gradient Boosting Classifier predicts the validation set with accuracy of: ",
          100 * (len(valid_predict) - wrong_answer) / (len(valid_predict)))

    test_predict = clf.predict(test_set[0])
    wrong_answer = 0
    for i, j in zip(test_predict, test_set[1]):
        if i != j:
            wrong_answer += 1
    print("Gradient Boosting Classifier predicts the test set with accuracy of: ",
          100 * (len(test_predict) - wrong_answer) / (len(test_predict)))
