from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
from pyedflib import highlevel
import tensorflow as tf
import scipy.io as sio
import numpy as np
import datetime
import csv
import os


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
        
        
class PhysionetDataset:
    
    def __init__(self, foldername="Data/"):
        
        self.fields = [] 
        self.rows = [] 

        with open(foldername + "subject-info.csv", 'r') as csvfile:
            csvreader = csv.reader(csvfile) 
            self.fields = next(csvreader)  
            for row in csvreader: 
                self.rows.append(row)   
            
        self.X = list()
        self.Y = list()
        for i in range(35):
            try:
                signals, signal_headers, header = highlevel.read_edf(foldername + "Subject" + str(i) + "_2.edf")
            except:
                signals, signal_headers, header = highlevel.read_edf(foldername + "Subject0" + str(i) + "_2.edf")
            self.X.append(tf.reshape(signals, (1, 21, -1, 1)))
            self.Y.append(self.Binary(np.ceil(float(self.rows[i][4]))))
        
        self.X = tf.concat(self.X, 0)
        self.Y = np.array(self.Y)
    
    def Binary(self, a):
        x = np.zeros([36])
        x[np.int(a)] = 1
        return x
        
        
class Encoder1(tf.keras.Sequential):
    
    def __init__(self, shape):
        
        super(Encoder1, self).__init__()
        self.add(tf.keras.layers.Conv2D(input_shape=shape, filters=4,
                                        kernel_size=(1, 2000), strides=(1, 100),
                                        padding='same', activation='relu'))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(1, 10), strides=(1, 2), padding='same'))

        self.add(tf.keras.layers.Conv2D(input_shape=(21, 155, 4), filters=8,
                                        kernel_size=(1, 50), strides=(1, 10),
                                        padding='same', activation='relu'))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(1, 10), strides=(1, 2), padding='same'))
        
        self.add(tf.keras.layers.Flatten(input_shape=[21, 8, 8]))
        self.add(tf.keras.layers.Dropout(rate=0.2))
        self.add(tf.keras.layers.Dense(1000, activation='sigmoid'))
        self.add(tf.keras.layers.Dropout(rate=0.2))
        self.add(tf.keras.layers.Dense(200, activation='sigmoid'))
        self.add(tf.keras.layers.Dropout(rate=0.2))
        self.add(tf.keras.layers.Dense(36, activation='sigmoid'))
        return
    

class DeepModel:
    
    def __init__(self, input_size, dataset):
        
        self.encoder = Encoder1(shape=input_size)
        self.weights = None
        self.dataset = dataset
        return
    
    def train(self, epochs=100, batch_size=7):
    
        lst = get_available_gpus()
        if '/device:GPU:0' in lst:
            tf.device('/device:GPU:0')
            print('GPU is activated')
        elif '/device:XLA_CPU:0' in lst:
            tf.device('/device:XLA_CPU:0')
            print('TPU is activated')
        else:
            print('CPU only available')
           
        self.encoder.compile(loss=tf.keras.losses.categorical_crossentropy, 
                             optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01), 
                             metrics=[tf.keras.metrics.CategoricalAccuracy()])
        
        self.encoder.fit(x=self.dataset.X, y=self.dataset.Y, validation_split=0.1,
                  batch_size=batch_size, epochs=epochs, shuffle=True)
        #            ,callbacks=[tensorboard_cb])         
        
        self.weights = self.encoder.weights[0]
        return
    
    def get_filters(self):
        
        return tf.squeeze(self.weights).numpy()
        
    
EEG_SHAPE = (21, 31000, 1)

dataset = PhysionetDataset(foldername="EEGMA/")
deepmodel = DeepModel(EEG_SHAPE, dataset)
deepmodel.train(100, 7)
f = deepmodel.get_filters()

# model1 = Encoder1(shape=EEG_SHAPE)
# model1.summary()
# print(model1(dataset.X[20:30, :, :]), '\n', dataset.Y[20:30])

# for i in range(4):
#   plt.subplot(4, 1, i+1)
#   plt.plot(tf.squeeze(model1.weights[0][:, :, :, i]).numpy())

# plt.show()
# # log_dir="logs1/" 
# # os.mkdir(log_dir)
# # tensorboard_cb = TensorBoard(log_dir=log_dir, update_freq='batch', histogram_freq=1)


    
