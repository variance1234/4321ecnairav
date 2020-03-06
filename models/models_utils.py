import os
import ntpath

from keras.callbacks import Callback
from keras import backend as K

import numpy as np

def get_script_folder(file):
    path = os.path.realpath(file)
    dir = ntpath.dirname(path)
    return dir

class LRLog(Callback):
    def on_epoch_end(self, epoch, logs={}):
        logs['lr'] = K.get_value(self.model.optimizer.lr)

def set_weights_to_randoms(model):
    layers = model.layers
    for i in range(len(layers)):
        w = model.layers[i].get_weights()
        ran_w = []
        for wm in w:
            ran_w.append(np.random.sample(wm.shape) - 0.5)
        model.layers[i].set_weights(ran_w)
