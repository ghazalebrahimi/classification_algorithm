from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def random_forest_classifier(train_set, valid_set, test_set):

    clf = RandomForestClassifier()
    clf.fit(train_set[0], train_set[1])

    valid_predict = clf.predict(valid_set[0])
    print ("Random Forest Classifier predicts the validation set with accuracy of:  :",
           accuracy_score(valid_predict, valid_set[1]))

    test_predict = clf.predict(test_set[0])
    print ("Random Forest Classifier predicts the test set with accuracy of: ",
           accuracy_score(test_predict, test_set[1]))
