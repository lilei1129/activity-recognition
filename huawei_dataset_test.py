
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from keras.models import *
from keras.layers import LSTM, Dense
import numpy as np
from keras.callbacks import*
import json
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

import os

data_dim = 20
timesteps = 300
num_classes = 8

##########################    预处理     ############################################

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
    return np.transpose(np.array(X_signals), (1, 2, 0))
    # Output classes to learn how to classify
LABELS = [
    "Still",
    "Walk",
    "Run",
    "Bike",
    "Car",
    "Bus",
    "Train",
    "Subway"
]

def pre_process(xfile,label,trainSavDir,num):
        file = open(xfile, 'r')
        for i in range(8):
            mkdir(trainSavDir+'/'+str(i))
        for i,row in enumerate(file):
            line=row.replace('  ', ' ').strip().split(' ')
            for j in range(num):
                jsonname = str(i)+'_'+str(j)+'_'+os.path.split(xfile)[1].replace('.txt','.json')
                with open(trainSavDir+'/'+str(int(label[i*39+j])-1)+'/'+jsonname,'w',encoding='utf-8') as jsonfile:
                    json.dump(line[0:300], jsonfile, ensure_ascii=False)


def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder...  ---")
		print ("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")

def load_y(y_path,sub_len):
    label=[]
    file = open(y_path, 'r')
    for row in file:
        line=[]
        for serie in row.split(' '):
            line.append(float(serie))
        #line=[np.array(serie, dtype=np.float32) for serie in [row.split(' ')]]
        for i in range(6000):
            if i%150==0 and (i+sub_len)<=6000:
                label.append(np.mean(line[i:i+sub_len]))
        #label.append([np.array(serie, dtype=np.float32) for serie in [row.split(' ')]])
        #print(label)
#        if(count>10):
#            break
#        if len(label)>200:
#            break
    file.close()
#    label2=np.array(label, dtype=np.int32)
#    label2=tf.reshape(label2,[-1,timesteps])
#    print(label2)
#    print(label2.shape)
  #  return label #tf.one_hot(label2,num_classes)
    return label         
trainSavDir='./save_dir'
mkdir(trainSavDir)
label=load_y('./all_data/Label.txt',300)
INPUT_SIGNAL_TYPES = [
    "Acc_x",
    "Acc_y",
    "Acc_z",
    "Gra_x",
    "Gra_y",
    "Gra_z",
    "Gyr_x",
    "Gyr_y",
    "Gyr_z",
    "Mag_x",
    "Mag_y",
    "Mag_z",
    "Ori_x",
    "Ori_y",
    "Ori_z",
    "Ori_w",
    "LAcc_x",
    "LAcc_y",
    "LAcc_z",
    "Pressure",
]
for file in INPUT_SIGNAL_TYPES:
    pre_process('./all_data/'+file+'.txt',label,trainSavDir,39)

######################    划分验证集     ################################
import shutil
import random

def split_for_Ver(json_dir,trian_dir,test_dir,percent):
    for i in range(8):
        mkdir(trian_dir+'/'+str(i))
    for i in range(8):
        mkdir(test_dir+'/'+str(i))
    image_list=[]
    for dir0 in os.listdir(json_dir):
        for file in os.listdir(json_dir+'/'+dir0):
            if'_Acc_x' in file:
                s=file.split('_')
                ss=s[0]+'_'+s[1]+'_'
                image_list.append(dir0+'/'+ss)

    random.shuffle(image_list)
    train_image_list=list(image_list[0:int(len(image_list)*(1-percent))])    
    test_image_list=list(image_list[int(len(image_list)*(1-percent)):len(image_list)])
    for filename in train_image_list:
        for types in INPUT_SIGNAL_TYPES:
            s=json_dir+'/'+filename+types+'.json'
            shutil.copy(s,'./train/'+filename+types+'.json')
    for filename in test_image_list:
        for types in INPUT_SIGNAL_TYPES:
            s=json_dir+'/'+filename+types+'.json'
            shutil.copy(s,'./test/'+filename+types+'.json')

split_for_Ver('./save_dir','./train','./test',0.3)

#
#TRAIN = "train/"
#TEST = "test/"
#X_train_signals_paths = [
#    TRAIN + signal + ".txt" for signal in INPUT_SIGNAL_TYPES
#]
#X_test_signals_paths = [
#    TEST + signal + ".txt" for signal in INPUT_SIGNAL_TYPES
#]
#y_train_path = TRAIN + "Label.txt"
#y_test_path = TEST + "y_test.txt"
#
#X_train = load_X(X_train_signals_paths)
#
#X_test = load_X(X_test_signals_paths)
#print('X_train',X_train.shape)
#
#
#y_train = load_y(y_train_path)
#y_test = load_y(y_test_path)
#X_train=tf.reshape(X_train,[-1,timesteps,data_dim])
#y_train=tf.reshape(y_train,[-1,timesteps,num_classes])
#
#
#print(X_train[0,0,:])

###########################    构建LSTM模型    ###############################################
model = Sequential()  
model.add(LSTM(32, return_sequences=True, stateful=False,batch_input_shape=(None, timesteps, data_dim)))
model.add(LSTM(64, return_sequences=True, stateful=False))
model.add(LSTM(64, return_sequences=True, stateful=False))
model.add(LSTM(32, return_sequences=True, stateful=False))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])


############################   训练    ###############################################
from keras.optimizers import Adam, SGD
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator

resultA=[]            
def generate_arrays_from_file(path,batch,width=300,height=20):
    train_list=[]
    for dir0 in os.listdir(path):
        for file in os.listdir(path+'/'+dir0):
            if'_Acc_x' in file:
                s=file.split('_')
                ss=s[0]+'_'+s[1]+'_'
                train_list.append(dir0+'/'+ss)
    random.shuffle(train_list)
    while True:
        for i in range(0,len(train_list),batch):
            xm_batch=train_list[i:i+batch]#文件索引
            y_batch=np.zeros(shape=(batch,num_classes),dtype=np.float32)
            for nnn,ss in enumerate(xm_batch):
                ssss=ss.split('/')[0]
                y_batch[nnn,:]=(np.eye(num_classes)[np.array(int(ssss), dtype=np.int32)])          
            x_batch=np.zeros(shape=(batch,width,height),dtype=np.float32)
            for k in range(batch):
                for j,types in enumerate(INPUT_SIGNAL_TYPES):
                    with open(path+'/'+xm_batch[k]+types+'.json','r',encoding='utf-8') as jsonfile:
                        serie=json.load(jsonfile)
                        m=np.array(serie, dtype=np.float32)
                        x_batch[k,:,j]=m[:]                    
            yield(x_batch,y_batch)

#generate_arrays_from_file('./save_dir',batch=5,width=300,height=20)





batch=20 
filepath=r"./h5/lstm01-{epoch:02d}-{val_acc:.4f}.h5"
checkpoint= ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list= [checkpoint]          
resultA = model.fit_generator(generate_arrays_from_file('./train',batch,width=300,height=20),
                              samples_per_epoch=int(len(label)*0.7)//batch*batch,
                              nb_epoch=200,
                              verbose=1,
                              callbacks=callbacks_list,
                              validation_data=generate_arrays_from_file('./test',batch,width=300,height=20),
                              nb_val_samples=(int(len(label)*0.3)//batch*batch),
                              class_weight='auto'
                              )
        
        


resultA.append(resultA)

plt.plot(resultA.history['acc'],'b')
plt.plot(resultA.history['val_acc'],'r')
plt.plot(resultA.history['loss'],'b')
plt.plot(resultA.history['val_loss'],'r')
plt.show()
model.save('models/Mode1_me2.h5') 



############################   预测    ###############################################

#X_test=X_test[0:2800]
#print(X_test.shape)
#y_test=y_test[0:2800]
#score = model.evaluate(X_test, y_test,batch_size=32)
#
#
#print()
#print ("Loss = " + str(score[0]))
#print ("Test Accuracy = " + str(score[1]))
#
#
