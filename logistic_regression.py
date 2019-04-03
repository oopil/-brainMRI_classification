import numpy as np
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from data import *

option_num = 0 # P V T options
class_num = 2
ford_num = 10
data, label = dataloader(class_num, option_num)
X,Y = shuffle_2arr(data, label)
X_train, Y_train, X_test, Y_test = split_train_test(X,Y, option_num, ford_num)
X_test, Y_test = valence_class(X_test, Y_test, class_num)
# X_train, Y_train = valence_class(X_train, Y_train, class_num)
print(X_train.shape, X_test.shape)

if class_num == 2:
    logreg = LogisticRegression(solver='lbfgs')
elif class_num == 3:
    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial') # multinomial / auto/ ovr
logreg.max_iter = 1000
# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, Y_train)

Pred = logreg.predict(X_test)
print('label\t:',Y_test)
print('predict :',Pred)
total_num = len(Y_test)
correct_answer = 0
for i in range(total_num):
    if Y_test[i] == Pred[i]:
        correct_answer += 1

print('the probability is ')
prob = correct_answer*100 / total_num
print(prob)


# assert False
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
# y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
# print(x_min)
# h = .02  # step size in the mesh
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# print(xx.ravel())
# Pred = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Pred = Pred.reshape(xx.shape)
# plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(xx, yy, Pred, cmap=plt.cm.Paired)
#
# # Plot also the training points
# plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
#
# plt.show()
