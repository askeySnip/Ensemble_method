import Adaboost
from sklearn.datasets import make_hastie_10_2
from sklearn.cross_validation import train_test_split
import pandas as pd

adaboost = Adaboost_classifier()
x, y = make_hastie_10_2()
df = pd.DataFrame(x)
df['Y'] = y

train, test = train_test_split(df, test_size = 0.2)
x_train, y_train = train.ix[:, :-1], train.ix[:,-1]
x_test, y_test = test.ix[:, :-1], test.ix[:, -1]

adaboost.fit(x_train, y_train);
adaboost.predict(x_train)
