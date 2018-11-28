from Adaboost import *
from sklearn.datasets import make_hastie_10_2
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

if __name__ == '__main__':
    clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
    adaboost = Adaboost_classifier(40, clf_tree)
    x, y = make_hastie_10_2()
    df = pd.DataFrame(x)
    df['Y'] = y

    train, test = train_test_split(df, test_size = 0.2)
    x_train, y_train = train.ix[:, :-1], train.ix[:,-1]
    x_test, y_test = test.ix[:, :-1], test.ix[:, -1]

    adaboost.fit(x_train, y_train);
    pred = adaboost.predict(x_train)
    print(adaboost.get_error_rate(pred, y_train))
