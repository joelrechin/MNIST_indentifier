import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.models import Sequential
#from bayes_opt import BayesianOptimization
from functools import partial
from hyperopt import hp, fmin, tpe
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, make_scorer
from keras.datasets import mnist
from keras.utils import to_categorical

#Function to show images.
def show_nums():
    (X1,y1),(X2,y2) = keras.datasets.mnist.load_data()
    fig, axes = plt.subplots(10,10,figsize=(8,8),subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1,wspace=0.1)) 
    for i, ax in enumerate(axes.flat):
        ax.imshow(X1[i],cmap='binary', interpolation='nearest')
        ax.text(0.05,0.05,str(y1[i]),transform=ax.transAxes,color='red')
    plt.show()

def CNN_data():
    (Xtrain,ytrain),(Xtest,ytest) = keras.datasets.mnist.load_data()
    Xtrain = Xtrain.reshape((60000,28,28,1))
    Xtest = Xtest.reshape((10000,28,28,1))
    Xtrain = Xtrain.astype('float32')
    Xtest = Xtest.astype('float32')
    Xtrain /= 255 
    Xtest /= 255
    ytest = ytest.astype(np.int32)
    ytrain = ytrain.astype(np.int32)
    y_train = keras.utils.to_categorical(ytrain,num_classes=10)
    y_test = keras.utils.to_categorical(ytest,num_classes=10)
    return Xtrain, Xtest, ytrain, ytest

def FFNN_data():
    (Xtrain,ytrain),(Xtest,ytest) = keras.datasets.mnist.load_data()
    Xtrain = Xtrain.reshape(60000, 784) 
    Xtest = Xtest.reshape(10000, 784)
    Xtrain = Xtrain.astype('float32')
    Xtest = Xtest.astype('float32')
    Xtrain /= 255 
    Xtest /= 255
    ytest = ytest.astype(np.int32)
    ytrain = ytrain.astype(np.int32)
    y_train = keras.utils.to_categorical(ytrain,num_classes=10)
    y_test = keras.utils.to_categorical(ytest,num_classes=10)
    return Xtrain, Xtest, ytrain, ytest

#CNN Model
def CNN_model(filters=(5,5),strides=(1,1)):
    Xtrain, Xtest, ytrain, ytest = CNN_data()
    model = Sequential()
    model.add(Conv2D(20,filters,activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=strides))
    model.add(Conv2D(50,filters,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=strides))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

#FFNN Model
def FFNN_model(optimizer='adam',neurons=32):
    Xtrain, Xtest, ytrain, ytest = FFNN_data()
    model = Sequential()
    model.add(Dense(neurons,input_shape=(784,),name='dense_layer', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(neurons*2,name='dense_layer_2',activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10,name='dense_layer_3',activation='softmax'))
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #history = model.fit(Xtrain,ytrain,batch_size=128,epochs=5,verbose=1,validation_split=0.3,)
    #score = model.evaluate(Xtest,ytest,verbose=1)
    #print('\nTest Score: ', score[0])
    #print('\nTest Accuracy: ', score[1])
    return model
    
def param_FFNN():
    Xtrain, Xtest, ytrain, ytest = FFNN_data()
    param_space_FFNN = {
        'optimizer': ['SGD', 'Adam','RMSProp'],
        'neurons': [32,64,128]
        }
    model = KerasClassifier(build_fn=FFNN_model, epochs=50, batch_size=1, verbose=1)
    classifier = GridSearchCV(model, param_grid = param_space_FFNN, cv=2, verbose=True, n_jobs=-1)
    classifier.fit(Xtrain, ytrain)
    print(pd.DataFrame(classifier.cv_results_))
    print(classifier.best_params_)
    print(classifier.best_score_)
    print(classifier.best_estimator_)
    return classifier

def param_CNN():
    Xtrain, Xtest, ytrain, ytest = CNN_data()
    model = KerasClassifier(build_fn=CNN_model, epochs=50, batch_size=1, verbose=1)
    param_space_CNN = {
        'filters': [10,20,30],
        'strides': [(1,1),(2,2),(3,3)]
        }
    classifier = GridSearchCV(model, param_grid = param_space_CNN, cv=2, verbose=True, n_jobs=-1)
    classifier.fit(Xtrain, ytrain)
    print(pd.DataFrame(classifier.cv_results_))
    print(classifier.best_params_)
    print(classifier.best_score_)
    print(classifier.best_estimator_)
    

#LogisticRegression Model
def LR():
    Xtrain, Xtest, ytrain, ytest = FFNN_data()
    lr = LogisticRegression(max_iter=1000)
    #lr.fit(Xtrain,ytrain)
    #score = lr.score(Xtest, ytest)
    #predictions = lr.predict(Xtest)
    #return score, predictions
    param_space_LR = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.1,0.5,1.0]
        }
    classifier = GridSearchCV(lr, param_grid = param_space_LR, cv=2, verbose=True, n_jobs=-1)
    classifier.fit(Xtrain, ytrain)
    print(pd.DataFrame(classifier.cv_results_))
    print(classifier.best_params_)
    print(classifier.best_score_)
    print(classifier.best_estimator_)
    return model

show_nums()
param_CNN()
    





