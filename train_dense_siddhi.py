#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Importing  packages
import tensorflow as tf
from tensorflow import keras
import numpy
import os
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D,BatchNormalization, concatenate, AveragePooling2D
from keras.optimizers import Adam
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from math import floor
#from bayes_opt import BayesianOptimization
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report, roc_auc_score, cohen_kappa_score, f1_score, make_scorer
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras import regularizers, layers
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import RepeatedKFold
from pickle import load
from numpy import array, argmax
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding, BatchNormalization, concatenate, LSTM, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os, sys, itertools, numpy as np, tensorflow as tf, tensorflow, numpy
from tensorflow.keras.constraints import MaxNorm as maxnorm
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from multiprocessing import Pool
from keras.utils import np_utils
from collections import Counter
from tensorflow.keras.utils import pad_sequences
from sklearn.metrics import roc_curve, auc, average_precision_score
from keras.models import load_model
import math
from sklearn import metrics


print("Library Loaded")

def comparison(testlabel, resultslabel):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for row1 in range(len(resultslabel)):
        if resultslabel[row1] < 0.5:
            resultslabel[row1] = 0
        else:
            resultslabel[row1] = 1
    for row2 in range(len(testlabel)):
        if testlabel[row2] == 1 and testlabel[row2] == resultslabel[row2]:
            TP = TP + 1
        if testlabel[row2] == 0 and testlabel[row2] != resultslabel[row2]:
            FP = FP + 1
        if testlabel[row2] == 0 and testlabel[row2] == resultslabel[row2]:
            TN = TN + 1
        if testlabel[row2] == 1 and testlabel[row2] != resultslabel[row2]:
            FN = FN + 1
    if TP + FN != 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 0
    if TN + FP != 0:
        TNR = TN / (TN + FP)
    else:
        TNR = 0
    if TP + FP != 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 0
    if TN + FN != 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 0
    if FN + TP != 0:
        FNR = FN / (FN + TP)
    else:
        FNR = 0
    if FP + TN != 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 0
    if FP + TP != 0:
        FDR = FP / (FP + TP)
    else:
        FDR = 0
    if FN + TN != 0:
        FOR = FN / (FN + TN)
    else:
        FOR = 0
    if TP + TN + FP + FN != 0:
        ACC = (TP + TN) / (TP + TN + FP + FN)
    else:
        ACC = 0
    if TP + FP + FN != 0:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    else:
        F1 = 0
    if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
        MCC = (TP * TN + FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC = 0
    if TPR != 0 and TNR != 0:
        BM = TPR + TNR - 1
    else:
        BM = 0
    if PPV != 0 and NPV != 0:
        MK = PPV + NPV - 1
    else:
        MK = 0
    return TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK



def mono_hot_encode(seq):
	print(seq)
	mapping = dict(zip(['A','T','G','C'], range(4)))  
	'''print(mapping)'''
	text=[seq[i:i+1] for i in range(len(seq)-(1-1))]
	'''print(text)'''
	seq1 = [mapping[i] for i in text]
	'''print(seq1)'''
	return np.eye(4)[seq1]


def kmers(file1,k):
	'''print(file1,k)'''
	seq_id1 = []
	kmer1 = []
	for line in file1:
		'''print(line)'''
		if line.startswith(">"):
			seq_id = line.strip()
			'''print(seq_id)'''
		else:
			if(len(line) >= 400):
			   '''print(line)'''
				for i in range(0,len(line)-k+1):
					'''print(i)'''
					kmer = line[i:i+k]
					'''print(kmer)'''
					seq = `seq_id + "_" + str(i)
					'''print(seq)'''
					seq_id1.append(seq)
					'''print(seq_id1)'''
					kmer1.append(kmer)
					'''print(kmer1)'''
			elif(len(line) >= 200):
				'''print(line)'''
				seq_id1.append(seq_id)
				'''print(seq_id1)'''
				kmer1.append(line)
				'''print(kmer1)'''
	return(seq_id1,kmer1)

def s1(k):
	'''print(k)'''
	texts_mono_fold = []
	texts_mono_fold.append(mono_hot_encode1(test12[k]))
	'''print(texts_mono_fold)'''
	padded_docs3 = pad_sequences(texts_mono_fold, maxlen=400, padding='post') 
	'''print (padded_docs3)'''
	texts_mono = []
	lab = testlab[k]
	'''print(lab)'''
	texts_mono.append(mono_hot_encode(testseq[k]))
	'''print(texts_mono)'''
	padded_docs4 = pad_sequences(texts_mono, maxlen=400, padding='post')
	'''print(padded_docs4)'''
	return lab, padded_docs4, padded_docs3



def mono_hot_encode1(seq):
	'''print(seq)'''
	mapping = dict(zip(['.',')','('], range(3)))  
	'''print(mapping)'''
	text=[seq[i:i+1] for i in range(len(seq)-(1-1))]
	'''print(text)'''
	seq1 = [mapping[i] for i in text]
	'''print(seq1)'''
	return np.eye(3)[seq1]




print("Function Loaded")

#Uploading our training file
filename=str("train2")
testseq=[]
testlab=[];test12=[]
for i in open(filename):
    '''print(i)'''
	z=i.split()
	testlab.append(int(z[0]))
	testseq.append(str(z[1]))
	test12.append(z[2])


name=[]
name = Pool().map(s1, [sent for sent in range(len(testseq))])
input_ids=[]
attention_masks=[]
attention_masks_fold=[]
for i, j in enumerate(name):
    '''print(i, j)'''
	input_ids.append(name[i][0])	
	attention_masks.append(name[i][1])
	attention_masks_fold.append(name[i][2])

train_inp = numpy.array(attention_masks).reshape(len(numpy.array(attention_masks)),400,4,1);attention_masks=[]
train_label = numpy.array(input_ids);input_ids=[]
train_fold = numpy.array(attention_masks_fold).reshape(len(numpy.array(attention_masks_fold)),400,3,1);attention_masks_fold=[]

#Uploading the testing file
filename=str("test2")
testseq=[]
testlab=[];test12=[]
for i in open(filename):
    '''print(i)'''
	z=i.split()
	testlab.append(int(z[0]))
	testseq.append(str(z[1]))
	test12.append(z[2])


name=[]
name = Pool().map(s1, [sent for sent in range(len(testseq))])
input_ids=[]
attention_masks=[]
attention_masks_fold=[]
for i, j in enumerate(name):
    '''print(i, j)'''
	input_ids.append(name[i][0])	
	attention_masks.append(name[i][1])
	attention_masks_fold.append(name[i][2])

test_inp = numpy.array(attention_masks).reshape(len(numpy.array(attention_masks)),400,4,1);attention_masks=[]
test_label = numpy.array(input_ids);input_ids=[]
test_fold = numpy.array(attention_masks_fold).reshape(len(numpy.array(attention_masks_fold)),400,3,1);attention_masks_fold=[]
'''
#Importing packages for densenet
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D,BatchNormalization, concatenate, AveragePooling2D
from keras.optimizers import Adam

def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)
    return conv_x

def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate
        return block_x, filters
        
        

dense_block_size = 10
layers_in_block = 4

growth_rate = 16
classes = 1
### One input
input_img = Input(shape=(400, 4, 1))
conv1 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(input_img)
batch1 = BatchNormalization()(conv1)
dense_x = MaxPooling2D((2, 2),strides=(2, 2),padding='same')(batch1)
#dense_x = Dropout(0.2)(dense_x)


for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, growth_rate * 2, growth_rate, 4)
        #dense_x, filters = transition_block(dense_x, filters)
        trans_x = BatchNormalization()(dense_x)
        trans_x = Activation('relu')(trans_x)
        trans_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
        batch1 = BatchNormalization()(trans_x)
        trans_x = MaxPooling2D((1, 1), strides=(2, 2))(batch1)
        dense_x, filters=trans_x, filters
	
dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)

### second input        
input_img2 = Input(shape=(400, 3, 1))
conv2 = Conv2D(32, (3, 3), padding='same', activation='selu', kernel_constraint=maxnorm(3))(input_img2)
batch3 = BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(batch3)
conv3 = Conv2D(64, (3, 3), padding='same', activation='selu', kernel_constraint=maxnorm(3))(pool2)
batch4 = BatchNormalization()(conv3)
dense_y = MaxPooling2D(pool_size=(2, 2),padding='same')(batch4)
#dense_y = Dropout(0.2)(dense_y)

### dense net
for block in range(dense_block_size - 1):
        dense_y, filters = dense_block(dense_y, growth_rate * 2, growth_rate, 4)
        #dense_x, filters = transition_block(dense_y, filters)
        trans_x = BatchNormalization()(dense_y)
        trans_x = Activation('relu')(trans_x)
        trans_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
        batch1 = BatchNormalization()(trans_x)
        trans_x = MaxPooling2D((1, 1), strides=(2, 2))(batch1)
        dense_y, filters3 =trans_x, filters

### dense block
dense_y, filters = dense_block(dense_y, filters, growth_rate, layers_in_block)
mode = concatenate([dense_x,dense_y], axis=-1)
dense_z = BatchNormalization()(mode)
dense_z = Activation('relu')(mode)
dense_z = GlobalAveragePooling2D()(dense_z)
output = Dense(classes, activation='exponential')(dense_z)
model=Model([input_img,input_img2], output)
#model.summary()


# training
batch_size = 100
epochs = 50
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer = optimizer, loss = 'mean_absolute_error', metrics=['accuracy'])
model.fit([train_inp,train_fold], train_label, epochs=epochs, batch_size=batch_size, shuffle=True,validation_data=([test_inp,test_fold], test_label))

#SAVING THE MODEL AND LOADING IT
model.save("train2")

model = load_model("train2")
y_pred = model.predict([test_inp, test_fold], verbose=1)
for j,i in enumerate(y_pred):
	print(train_label[j],i)
fpr, tpr, threshold = roc_curve(test_label,y_pred)
te_roc_auc = auc(fpr,tpr)
f3 = open("performance_metrices",'w')
f3.writelines("TP"+"\t"+"TN"+"\t"+"FN"+"\t"+"FP"+"\t"+"TPR"+"\t"+"TNR"+"\t"+"ACC"+"\t"+"F1"+"\t"+"MCC"+"\t"+"auc"+"\n")

TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK = comparison(test_label,y_pred)

f3.writelines(str(TP)+"\t"+str(TN)+"\t"+str(FN)+"\t"+str(FP)+"\t"+str(TPR)+"\t"+str(TNR)+"\t"+str(ACC)+"\t"+str(F1)+"\t"+str(MCC)+"\t"+str(te_roc_auc)+"\n")
f3.close()


###Traning and Testing Visualisation

import sys
import matplotlib
print("Generating plots...")
sys.stdout.flush()
matplotlib.use("Agg")
matplotlib.pyplot.style.use("ggplot")
matplotlib.pyplot.figure()
N = epochs 
matplotlib.pyplot.plot(np.arange(0, N), history.history["loss"], label="train_loss")
matplotlib.pyplot.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
matplotlib.pyplot.plot(np.arange(0, N), history.history["acc"], label="train_acc")
matplotlib.pyplot.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
matplotlib.pyplot.title("Cactus Image Classification")
matplotlib.pyplot.xlabel("Epoch #")
matplotlib.pyplot.ylabel("Loss/Accuracy")
matplotlib.pyplot.legend(loc="lower left")
matplotlib.pyplot.savefig("plot.png")



###Testing
from sklearn import metrics
label_pred = model.predict(X_test)

pred = []
for i in range(len(label_pred)):
    pred.append(np.argmax(label_pred[i]))

Y_test = np.argmax(Cat_test_y, axis=1) # Convert one-hot to index

print(metrics.classification_report(Y_test, pred))
from sklearn import metrics
label_pred = model.predict(X_test)

pred = []
for i in range(len(label_pred)):
    pred.append(np.argmax(label_pred[i]))

Y_test = np.argmax(Cat_test_y, axis=1) # Convert one-hot to index

print(metrics.accuracy_score(Y_test, pred))
'''
Here's the code of the project, use .py to run the script