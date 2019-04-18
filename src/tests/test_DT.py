import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
import pydotplus

pima = pd.read_csv("diabetes.csv")
col_names = pima.columns.tolist()
print(col_names)

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age', 'DiabetesPedigreeFunction']
X = pima[feature_cols]
print("X shape = {0}".format(X.shape))
Y = pima['Outcome']
print("Y shape = {0}".format(Y.shape))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
print("X_train shape = {0}".format(X_train.shape))
print("Y_train shape = {0}".format(Y_train.shape))
print("X_test shape = {0}".format(X_test.shape))
print("Y_test shape = {0}".format(Y_test.shape))

clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

test_score = clf.score(X_test, Y_test)
print("Testing accuracy = {0}, {1}".format(test_score,metrics.accuracy_score(Y_test,clf.predict(X_test))))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = feature_cols, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('unpruned_diabetes.png')

clf1 = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=3)
clf1.fit(X_train, Y_train)

test_score1 = clf1.score(X_test, Y_test)
print("Testing accuracy of pruned model = {0}, {1}".format(test_score1,metrics.accuracy_score(Y_test,clf1.predict(X_test))))

dot_data1 = StringIO()
export_graphviz(clf1, out_file=dot_data1, filled=True, rounded=True, special_characters=True, feature_names = feature_cols, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('pruned_diabetes.png')
