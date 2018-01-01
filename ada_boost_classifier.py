import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from color import color


def ada_boost_classifier(train_set, valid_set, test_set):
    start_time = time.time()

    clf = AdaBoostClassifier(n_estimators=50)
    clf.fit(train_set[0], train_set[1])

    valid_predict = clf.predict(valid_set[0])
    print ("Ada Boost Classifier predicts the validation set with accuracy of:  :",
           accuracy_score(valid_predict, valid_set[1]))

    test_predict = clf.predict(test_set[0])
    print (color.GREEN + "Ada Boost Classifier predicts the test set with accuracy of: ",
           accuracy_score(test_predict, test_set[1]), color.END)

    end_time = time.time()
    print(color.RED + "Total time in seconds: ", end_time - start_time, color.END)


'''
n_estimators = 10 ==> Ada Boost Classifier predicts the validation set with accuracy of:  : 0.588

real    0m16.129s
user    0m15.838s
sys     0m0.275s

'''

'''
n_estimators = 20 ==> Ada Boost Classifier predicts the validation set with accuracy of:  : 0.6856

real    0m31.007s
user    0m30.249s
sys     0m0.460s

'''

'''
n_estimators = 50 ==> Ada Boost Classifier predicts the validation set with accuracy of:  : 0.744

real    1m6.766s
user    1m5.701s
sys     0m0.637s

'''

'''
n_estimators = 100 ==> Ada Boost Classifier predicts the validation set with accuracy of:  : 0.7263

real    2m18.322s
user    2m16.508s
sys     0m1.046s

'''

'''
n_estimators = 75 ==> Ada Boost Classifier predicts the validation set with accuracy of:  : 0.7436

real    1m36.649s
user    1m36.162s
sys     0m0.378s

'''