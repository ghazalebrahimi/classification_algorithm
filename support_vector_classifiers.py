import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from color import color


def support_vector_classifiers(train_set, valid_set, test_set):
    start_time = time.time()

    clf = SVC(decision_function_shape='ovr')
    clf.fit(train_set[0][:10000], train_set[1][:10000])

    valid_predict = clf.predict(valid_set[0])
    print ("Support vector classifier predicts the validation set with accuracy of:  :",
           accuracy_score(valid_predict, valid_set[1]))

    test_predict = clf.predict(test_set[0])
    print (color.GREEN + "Support vector classifier predicts the test set with accuracy of: ",
           accuracy_score(test_predict, test_set[1]), color.END)

    middle_time = time.time()
    print(color.RED + "Total time in seconds: ", middle_time - start_time, color.END)

    clf = LinearSVC()
    clf.fit(train_set[0], train_set[1])

    valid_predict = clf.predict(valid_set[0])
    print ("Linear support vector classifier predicts the validation set with accuracy of:  :",
           accuracy_score(valid_predict, valid_set[1]))

    test_predict = clf.predict(test_set[0])
    print (color.GREEN + "Linear support vector classifier predicts the test set with accuracy of: ",
           accuracy_score(test_predict, test_set[1]), color.END)

    end_time = time.time()
    print(color.RED + "Total time in seconds: ", end_time - middle_time, color.END)
