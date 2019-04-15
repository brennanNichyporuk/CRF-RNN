import tensorflow as tf

import keras
import keras.backend as K
from keras import layers, models, metrics, callbacks

from keras_fcn.keras_fcn import FCN

import lattice_filter_op_loader
module = lattice_filter_op_loader.module

import skimage
import skimage.io
import scipy
import scipy.io

import numpy as np
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default='')
args = parser.parse_args()

class DataGenerator(keras.utils.Sequence):
    def __init__(self, root, txt_filename, mean, std, batch_size,
            dim=(500, 500), n_channels=3, n_classes=21, shuffle=True):
        with open('%s/%s' % (root, txt_filename), 'r') as f:
            self.image_names = f.read().split()
        self.root = root
        self.txt_filename = txt_filename
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.image_names)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        Y = np.zeros((self.batch_size, *self.dim, self.n_classes), dtype=int)
        for i, index in enumerate(indexes):
            x = skimage.io.imread('%s/img/%s.jpg' % (self.root, self.image_names[index])).astype(float)
            x = (x-self.mean) / self.std

            y = scipy.io.loadmat('%s/cls/%s.mat' % (self.root, self.image_names[index]))['GTcls'][0]['Segmentation'][0]
            y = np.eye(self.n_classes)[y]
            
            h, w, c = x.shape
            h_min = math.floor((self.dim[0] - h) / 2)
            w_min = math.floor((self.dim[1] - w) / 2)
            
            X[i, h_min:(h_min+h), w_min:(w_min+w), :] = x
            Y[i, h_min:(h_min+h), w_min:(w_min+w), :] = y
        return X, Y 
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

##########
"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana , Miguel Monteiro, Walter de Back

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR

IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
class CRF_RNN(keras.layers.Layer):
    """ Implementation of:
    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015

    Based on: https://github.com/sadeepj/crfasrnn_keras/blob/master/src/crfrnn_layer.py
    and https://github.com/MiguelMonteiro/CRFasRNNLayer
    """
    
    def __init__(self, output_dim, num_iterations, theta_alpha, theta_beta, theta_gamma, **kwargs):
        self.output_dim = output_dim
        self.num_iterations = num_iterations
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        super(CRF_RNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.K0_weights = self.add_weight(name='K0_weights',
                                          shape=(self.output_dim[-1],),
                                          initializer=keras.initializers.Constant(value=1.0),
                                          trainable=True)
        self.K1_weights = self.add_weight(name='K1_weights',
                                          shape=(self.output_dim[-1],),
                                          initializer=keras.initializers.Constant(value=1.0),
                                          trainable=True)
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.output_dim[-1],self.output_dim[-1]),
                                                    initializer=keras.initializers.Identity(gain=-1.0),
                                                    trainable=True)
        super(CRF_RNN, self).build(input_shape)

    def call(self, x):
        # Unaries / Image
        I = x[0]
        U = x[1]
        
        Q = U
        for _ in range(self.num_iterations):
            # Normalizing
            Q = K.softmax(Q)

            # Message Passing
            Q0 = module.lattice_filter(Q, I, bilateral=False, theta_gamma=self.theta_gamma)
            Q1 = bilateral_out = module.lattice_filter(Q, I, bilateral=True, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta)
            
            # Weighting Filter Outputs
            Q = Q0*self.K0_weights + Q1*self.K1_weights

            # Compatibility Transform
            Q = K.dot(Q, self.compatibility_matrix) 

            # Adding Unary Potentials
            Q = U - Q
        return Q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.output_dim)
##########

def iou(y_true, y_pred, label):
    y_true = K.cast(K.equal(K.argmax(y_true, axis=-1), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred, axis=-1), label), K.floatx())
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection, union

def mean_iou(y_true, y_pred):
    num_classes = 0.0
    total_iou = K.variable(0)
    for i in range(21):
        intersection, union = iou(y_true, y_pred, i)
        total_iou += K.switch(K.equal(union, 0.0), 0.0, intersection / union)
        num_classes += K.switch(K.equal(union, 0.0), 0.0, 1.0)
    return total_iou / num_classes

inputs = layers.Input(shape=(500, 500, 3))
outputs = FCN(inputs,
              classes=21,
              weights='imagenet',
              trainable_encoder=True)
outputs = CRF_RNN((500, 500, 21), 5, 160., 3., 3.)([inputs,outputs])
outputs = layers.Activation('softmax')(outputs)
model = models.Model(inputs=inputs, outputs=outputs)

if args.resume:
    model.load_weights(args.resume, by_name=True)

optimizer = keras.optimizers.Adam(lr=1e-5, beta_1=0.99)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=[mean_iou, 'categorical_accuracy'])
model.summary()

root = '../datasets/benchmark_RELEASE/dataset'
mean = np.array([116.483, 112.998, 104.116])
std = np.array([60.429, 59.488, 60.941])
batch_size = 4

train_generator = DataGenerator(root, 'train_custom.txt', mean, std, batch_size)
val_generator = DataGenerator(root, 'val_custom.txt', mean, std, batch_size)

checkpointer = callbacks.ModelCheckpoint(filepath='./weights.hdf5', monitor='val_mean_iou', mode='max', verbose=1)
lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_mean_iou', min_delta=0.001, patience=6, verbose=1, mode='max')
early_stopping = callbacks.EarlyStopping( monitor='val_mean_iou', min_delta=0.001, patience=15, verbose=1, mode='max')
model.fit_generator(train_generator, validation_data=val_generator, epochs=240, callbacks=[checkpointer, lr_schedule, early_stopping])
