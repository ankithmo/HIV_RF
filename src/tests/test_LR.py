from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

digits = load_digits()
print "X shape = ", digits.data.shape
print "Y shape = ", digits.target.shape

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
print "X_train shape = ", x_train.shape
print "Y_train shape = ", y_train.shape
print "X_test shape = ", x_test.shape
print "Y_test shape = ", y_test.shape

LR = LogisticRegression()
LR.fit(x_train, y_train)

predictions = LR.predict(x_test)

score = LR.score(x_test, y_test)
print "Accuracy =", score 

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()
