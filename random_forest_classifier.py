import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from color import color


def random_forest_classifier(train_set, valid_set, test_set):
    start_time = time.time()

    clf = RandomForestClassifier(n_estimators=25, min_samples_split=10)
    clf.fit(train_set[0], train_set[1])

    valid_predict = clf.predict(valid_set[0])
    print ("Random Forest Classifier predicts the validation set with accuracy of:  :",
           accuracy_score(valid_predict, valid_set[1]))

    test_predict = clf.predict(test_set[0])
    print (color.GREEN + "Random Forest Classifier predicts the test set with accuracy of: ",
           accuracy_score(test_predict, test_set[1]), color.END)

    end_time = time.time()
    print(color.RED + "Total time in seconds: ", end_time - start_time, color.END)