import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier


digits=load_digits()

clf=MLPClassifier(hidden_layer_sizes=(200,),
        activation='logistic',solver='adam',
        learning_rate_init=0.0001,max_iter=40000
        )
print(clf)
clf.fit(digits.data[:1001],digits.target[:1001])

res=clf.predict(digits.data[1001:])
error_num=0
num=len(digits.data[1001:])
for i in range(num):
    if res[i]!=digits.target[i+1001]:
        error_num+=1
print(res[:5])
print("accuracy:",(num-error_num)/float(num))
