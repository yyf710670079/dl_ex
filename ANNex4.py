from keras.models import Sequential
from keras import optimizers, initializers, callbacks
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils

##加载数据，从sklearn中加载的mnist手写集
digits=load_digits()


##每层的初始化方法
initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)  #优化器设置
'''
def step_decay(epoch):
    if epoch<23:
        lrate=0.001
    elif epoch<40:
        lrate=0.00005
    else:
        lrate=0.00001
    #lrate = initial_lrate * math.pow(drop, epoch)
    return lrate
lrate = callbacks.LearningRateScheduler(step_decay)
'''
##多分类任务将y变为one-hot vector
y_train=np_utils.to_categorical(digits.target[:1500],num_classes=10)
y_test=np_utils.to_categorical(digits.target[1500:],num_classes=10)

##Adam优化算法
Adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  ##RMSprop+momentum


##模型构造
model = Sequential()
model.add(Dense(units=100, activation='relu',\
                kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None), input_dim=64))
model.add(Dropout(0.5))
model.add(Dense(units=100, activation='relu',\
                kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

##模型编译
model.compile(loss='categorical_crossentropy',
                            optimizer='Adam',
                            metrics=['accuracy'])

##模型输入实际数据
model.fit(digits.data[:1500], y_train, epochs=100, batch_size=32)
#weights=model.get_weights()

##记录模型权重到一个文件中
model.save_weights('my_model.h5')

##性能评估（mae 和 acc）
loss_and_metrics = model.evaluate(digits.data[1500:], y_test, batch_size=32)
print(loss_and_metrics)

##单个数据预测
preds= model.predict_classes(digits.data[1].reshape(1, 64))
print(preds)


##单个数据的可视化
image_1=digits.data[1].reshape(8, 8)
plt.imshow(image_1, cmap='gray')     ##灰度图
plt.show()