from sklearn import tree


class Classifier:
    features = [[4],[5],[12],[3]]
    lables = ["Rechteck","Pentagon","Kreis","Dreieck"]

    #just corner input
    def piceClassifier(self,corner):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(Classifier.features, Classifier.lables)
        return clf.predict([[corner]])
