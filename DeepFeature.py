from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
from pyedflib import highlevel
from scipy import signal

import tensorflow as tf
import scipy.io as sio
import numpy as np
import datetime

import zipfile
import wget
import csv
import os




def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
        
        
class SampleDataset:
    def __init__(self):
        
        self.X = None
        self.Y = None
        return
    
    
class PhysionetDataset:
    
    def __init__(self, foldername="EEGMA/", filter_order=2, lbf=10, ubf=20,
                 sampling_rate=500, download=False, download_colab=False):
        
        # super.__init__(self)
        self.fields = [] 
        self.rows = [] 
        if download:
            print("Downloading...")
            self.download(download=download_colab)
        
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
            nyq = sampling_rate/2
            b, a = signal.butter(filter_order, [lbf/nyq, ubf/nyq], btype='band')
            for k in range(21):
                signals[k, :] = signal.lfilter(b, a, signals[k, :])
            self.X.append(tf.reshape(signals, (1, 21, -1, 1)))
            self.Y.append(self.Binary(np.ceil(float(self.rows[i][4]))))
        
        self.X = tf.concat(self.X, 0)
        self.Y = np.array(self.Y)
    
    def Binary(self, a):
        x = np.zeros([36])
        x[np.int(a)] = 1
        return x
    
    def unzip(self, filename, folder):
        with zipfile.ZipFile(filename) as f:
            f.extractall(folder)
            
        print("File extraction complete.")
        return

    def download(self, url="https://www.physionet.org/static/published-projects/eegmat/eeg-during-mental-arithmetic-tasks-1.0.0.zip", download=True):
        if download:
            fname = wget.download(url)
        else:
            fname = "/content/eeg-during-mental-arithmetic-tasks-1.0.0.zip"
        
        try:
            os.mkdir('EEGMA')
        except:
            pass
        
        self.unzip(fname, 'EEGMA')
        print("Download complete.")
        return 
    
    
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
    
    
class Encoder2(tf.keras.Sequential):
    
    def __init__(self, shape):
        
        super(Encoder2, self).__init__()
        # self.add(tf.keras.layers.Flatten(input_shape=shape))
        # self.add(tf.keras.layers.Dropout(rate=0.5))
        self.add(tf.keras.layers.Dense(80, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        # self.add(tf.keras.layers.Dropout(rate=0.4))
        # self.add(tf.keras.layers.Dense(60, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        # self.add(tf.keras.layers.Dropout(rate=0.4))
        self.add(tf.keras.layers.Dense(20, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        # self.add(tf.keras.layers.Dropout(rate=0.4))
        self.add(tf.keras.layers.Dense(5, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        # self.add(tf.keras.layers.Dropout(rate=0.5))
        # self.add(tf.keras.layers.Dense(12, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        # self.add(tf.keras.layers.Dropout(rate=0.5))
        # self.add(tf.keras.layers.Dense(5, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        # self.add(tf.keras.layers.Dropout(rate=0.5))
        self.add(tf.keras.layers.Dense(2, activation='sigmoid'))
        return
    

class DeepModel1:
    
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
                             optimizer=tf.keras.optimizers.sgd(learning_rate=0.1), 
                             metrics=[tf.keras.metrics.CategoricalAccuracy()])
        
        self.encoder.fit(x=self.dataset.X, y=self.dataset.Y, validation_split=0.1,
                  batch_size=batch_size, epochs=epochs, shuffle=True)
        #            ,callbacks=[tensorboard_cb])         
        
        self.weights = self.encoder.weights[0]
        return
    
    def get_filters(self):
        
        return tf.squeeze(self.weights).numpy()
        

class DeepModel2:
    
    def __init__(self, input_size=None, dataset=None):
        
        self.encoder = Encoder2(shape=input_size)
        self.weights = None
        self.dataset = dataset
        return
    
    def train(self, epochs=100, batch_size=7, val=0.2, lr=0.001):
    
        lst = get_available_gpus()
        if '/device:GPU:0' in lst:
            tf.device('/device:GPU:0')
            print('GPU is activated')
        elif '/device:XLA_CPU:0' in lst:
            tf.device('/device:XLA_CPU:0')
            print('TPU is activated')
        else:
            print('CPU only available')
           
        self.encoder.compile(loss=tf.keras.losses.binary_crossentropy, 
                             optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                             metrics=[tf.keras.metrics.BinaryAccuracy()])
        
        self.encoder.fit(x=self.dataset.X, y=self.dataset.Y, validation_split=val,
                  batch_size=batch_size, epochs=epochs, shuffle=True)
        #            ,callbacks=[tensorboard_cb])         
        
        self.weights = self.encoder.weights[0]
        return
    
    def get_filters(self):
        
        return tf.squeeze(self.weights).numpy()
    
