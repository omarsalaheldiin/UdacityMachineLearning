def NBAccuracy(features_train, labels_train, features_test, labels_test):
    
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train )
    pred = clf.predict(features_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, pred)
    return accuracy
