
from keras.models import Sequential
import numpy as np 
import tensorflow as tf
from sklearn.datasets import load_digits
from keras.layers import Dense
from keras.utils import np_utils
digits=load_digits()
y_train=np_utils.to_categorical(digits.target[:1000],num_classes=10)
y_test=np_utils.to_categorical(digits.target[1000:],num_classes=10)


model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=64))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
model.fit(digits.data[:1000], y_train, epochs=5, batch_size=32)



loss_and_metrics = model.evaluate(digits.data[1000:], y_test, batch_size=128)


