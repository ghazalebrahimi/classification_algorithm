from sklearn.neighbors import KNeighborsClassifier


def k_nearest_neighbors(train_set, valid_set, test_set):
    nbrs = KNeighborsClassifier(n_neighbors=4, algorithm='auto')
    print (len(train_set[0]))
    nbrs_fit = nbrs.fit(train_set[0][0:3000], train_set[1][0:3000])
    valid_predict = nbrs.predict(valid_set[0])
    wrong_answer = 0
    print ("here")
    for i, j in zip(valid_predict, valid_set[1]):
        if i != j:
            wrong_answer += 1
    print("K Neighbors  Classifier predicts the validation set with accuracy of: ",
          100 * (len(valid_predict) - wrong_answer) / (len(valid_predict)))
