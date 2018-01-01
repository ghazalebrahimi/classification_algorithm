from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def gradient_boosting_classifier(train_set, valid_set, test_set):

    clf = GradientBoostingClassifier()
    clf.fit(train_set[0][:1000], train_set[1][:1000])

    valid_predict = clf.predict(valid_set[0])
    print ("Gradient Boosting Classifier predicts the validation set with accuracy of:  :",
           accuracy_score(valid_predict, valid_set[1]))

    test_predict = clf.predict(test_set[0])
    print ("Gradient Boosting Classifier predicts the test set with accuracy of: ",
           accuracy_score(test_predict, test_set[1]))
