

"""
Stacked LSTM TensorFlow

The original LSTM model is comprised of a single hidden LSTM layer
followed by a standard feedforward output layer.

The Stacked LSTM is an extension to this model that has multiple
hidden LSTM layers where each layer contains multiple memory cells.


Stacking LSTM hidden layers makes the model deeper, more accurately
earning the description as a deep learning technique.

It is the depth of neural networks that is generally attributed to the
success of the approach on a wide range of challenging prediction problems.

```the success of deep neural networks is commonly attributed to the hierarchy
that is introduced due to the several layers. Each layer processes some part
of the task we wish to solve, and passes it on to the next. In this sense,
the DNN can be seen as a processing pipeline, in which each layer solves a
part of the task before passing it on to the next, until finally the last
layer provides the output.``` - "Training and Analysing Deep Recurrent Neural Networks"
by Michiel Hermans and Benjamin Schrauwen.


Additional hidden layers can be added to a Multilayer Perceptron neural network to
make it deeper. The additional hidden layers are understood to recombine the learned
representation from prior layers and create new representations at high levels of
abstraction. For example, from lines to shapes to objects.

A sufficiently large single hidden layer Multilayer Perceptron can be used to
approximate most functions. Increasing the depth of the network provides an alternate
solution that requires fewer neurons and trains faster. Ultimately, adding depth is a
type of representational optimization.

```Deep learning is built around a hypothesis that a deep, hierarchical model can be
exponentially more efficient at representing some functions than a shallow one.``` -
"How to Construct Deep Recurrent Neural Networks" by Razvan Pascanu, Caglar Gulcehre,
Kyunghyun Cho and Yoshua Bengio.


Reference-
https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

"Generating Sequences With Recurrent Neural Networks" by Alex Graves.
"""


# Specify GPU to be used-
%env CUDA_DEVICE_ORDER = PCI_BUS_ID
%env CUDA_VISIBLE_DEVICES = 0


import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.layers import LSTM, GRU, Input, Flatten, Dense, LSTM, RepeatVector
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Dropout
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


print(f"TensorFlow version: {tf.__version__}")
# TensorFlow version: 2.8.0

# Check GPU availibility-
gpu_devices = tf.config.list_physical_devices('GPU')
# print(f"GPU: {gpu_devices}")

if gpu_devices:
    print(f"GPU: {gpu_devices}")
    details = tf.config.experimental.get_device_details(gpu_devices[0])
    print(f"GPU details: {details.get('device_name', 'Unknown GPU')}")
else:
    print("No GPU found")
# No GPU found


"""
Stacked LSTM Architecture

The same benefits can be harnessed with LSTMs.

Given that LSTMs operate on sequence data, it means that the addition of
layers adds levels of abstraction of input observations over time.
In effect, chunking observations over time or representing the problem
at different time scales.

```… building a deep RNN by stacking multiple recurrent hidden states on top
of each other. This approach potentially allows the hidden state at each level
to operate at different timescale.``` - "How to Construct Deep Recurrent Neural
Networks".


Stacked LSTMs or Deep LSTMs were introduced by Graves, et al. in their application
of LSTMs to speech recognition, beating a benchmark on a challenging standard problem.

```RNNs are inherently deep in time, since their hidden state is a function of all
previous hidden states. The question that inspired this paper was whether RNNs could
also benefit from depth in space; that is from stacking multiple recurrent hidden
layers on top of each other, just as feedforward layers are stacked in conventional
deep networks.``` - "Speech Recognition With Deep Recurrent Neural Networks" by
Alex Graves, Abdel-rahman Mohamed and Geoffrey Hinton.

In the same work, they found that the depth of the network was more important than
the number of memory cells in a given layer to model skill.

Stacked LSTMs are now a stable technique for challenging sequence prediction problems.
A Stacked LSTM architecture can be defined as an LSTM model comprised of multiple LSTM
layers. An LSTM layer above provides a sequence output rather than a single value output
to the LSTM layer below (assuming the top down architecture) as: input -> LSTM -> LSTM ->
Dense -> output.

Specifically, one output per input time step, rather than one output time step for
all input time steps.
"""

# Each LSTMs memory cell requires a 3D input. When an LSTM processes one input
# sequence of time steps, each memory cell will output a single value for the
# whole sequence as a 2D array.

num_time_steps = 3
num_features = 1

# Define LSTM model where the LSTM layer is also the output layer-
model = Sequential()
model.add(
    LSTM(
        units = 1, input_shape = (num_time_steps, num_features)
        )
    )

# Compile defined model-
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 10e-3),
    loss = tf.keras.losses.MeanSquaredError
    )

# Define input data as: (batch_size, look_back_window_size, num_features)-
X = np.random.randn(1, num_time_steps, num_features)

# Get model's prediction using input data-
model.predict(X)
# array([[-0.14402393]], dtype=float32)

# LSTM model output shape: (batch_size, number of LSTMcells)-
model.predict(X).shape
# (1, 1)


'''
To stack LSTM layers, we need to change the configuration of the
prior LSTM layer to output a 3D array as input for the subsequent layer.

We can do this by setting the `return_sequences` argument on the layer to True
(defaults to False). This will return one output for each input time step and
provide a 3D array.
'''
# Define LSTM model where the LSTM layer is also the output layer-
model = Sequential()
model.add(
    LSTM(
        units = 1, input_shape = (num_time_steps, num_features),
        return_sequences = True
        )
    )

# Compile defined model-
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 10e-3),
    loss = tf.keras.losses.MeanSquaredError
    )

X.shape
# (1, 3, 1)

# Get hidden activation value for all time step in the input sequence-
model.predict(X).shape
# (1, 3, 1)

# Get model prediction-
model.predict(X)
'''
array([[[-0.12066168],
        [ 0.03475228],
        [ 0.07930516]]], dtype=float32)
'''


# We can continue to add hidden LSTM layers as long as the prior LSTM layer
# provides a 3D output as input for the subsequent layer.


class Stacked_LSTM(Model):
    '''
    Use-case for multivariate, multistep time-series prediction.
    '''
    def __init__(
        self, num_features,
        num_past_steps, num_future_steps
    ):
        super(Stacked_LSTM, self).__init__()
        self.num_features = num_features
        self.num_past_steps = num_past_steps
        self.num_future_steps = num_future_steps
        
        self.lstm1 = LSTM(
            units = 64, input_shape = (self.num_past_steps, self.num_features),
            return_sequences = True, activation = None
        )
        
        self.lstm2 = LSTM(
            units = 128, return_sequences = True,
            activation = None
        )
        
        self.lstm3 = LSTM(units = 32, activation = None)
        
        self.dense = Dense(units = self.num_future_steps, activation = None)
    
    
    def call(self, x):
        x = tf.nn.tanh(self.lstm1(x))
        x = tf.nn.tanh(self.lstm2(x))
        x = tf.nn.tanh(self.lstm3(x))
        x = self.dense(x)
        return x
    
    
    def shape_computation(self, x):
        print(f"Input shape: {x.shape}")
        x = tf.nn.tanh(self.lstm1(x))
        print(f"First LSTM layer output shape: {x.shape}")
        x = tf.nn.tanh(self.lstm2(x))
        print(f"Second LSTM layer output shape: {x.shape}")
        x = tf.nn.tanh(self.lstm3(x))
        print(f"Last LSTM layer output shape: {x.shape}")
        x = self.dense(x)
        print(f"Dense layer output shape: {x.shape}")
        return None


# Initialize Stacked LSTM architecture-
model = Stacked_LSTM(
    num_features = 8, num_past_steps = 50,
    num_future_steps = 10
)

# Define input as: (batch_size, look_back_window_size, num_features)-
X = np.random.randn(64, 50, 8)

X.shape
# (64, 50, 8)

model.shape_computation(X)
'''
Input shape: (64, 50, 8)
First LSTM layer output shape: (64, 50, 64)
Second LSTM layer output shape: (64, 50, 128)
Last LSTM layer output shape: (64, 32)
Dense layer output shape: (64, 10)
'''

# Get model predictions-
output = model(X)

# Output shape: (batch_size, num_future_steps)-
output.shape
# TensorShape([64, 10])


# Count number of trainable parameters in a layer-wise manner-
# Count number of trainable parameters in ResNet-18 model-
tot_params = 0

for layer in model.trainable_weights:
    loc_params = tf.math.count_nonzero(layer, axis = None).numpy()
    print(f"layer: {layer.shape} has {loc_params} params")
    tot_params += loc_params
'''
layer: (8, 256) has 2048 params
layer: (64, 256) has 16384 params
layer: (256,) has 64 params
layer: (64, 512) has 32768 params
layer: (128, 512) has 65536 params
layer: (512,) has 128 params
layer: (128, 128) has 16384 params
layer: (32, 128) has 4096 params
layer: (128,) has 32 params
layer: (32, 10) has 320 params
layer: (10,) has 0 params
'''

print(f"Stacked 3 layer LSTM model has {tot_params} trainable parameters")
# Stacked 3 layer LSTM model has 137760 trainable parameters

model.summary()
'''
Model: "stacked_lstm_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm_11 (LSTM)              multiple                  18688

 lstm_12 (LSTM)              multiple                  98816

 lstm_13 (LSTM)              multiple                  20608

 dense_1 (Dense)             multiple                  330

=================================================================
Total params: 138,442
Trainable params: 138,442
Non-trainable params: 0
_________________________________________________________________
'''

# METHOD-1: This also counts biases
trainable_wts = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_wts = np.sum([K.count_params(w) for w in model.non_trainable_weights])

print("\nNumber of training weights = {0} and non-trainabel weights = {1}\n".format(
    trainable_wts, non_trainable_wts
))
print("Total number of parameters = {0}\n".format(trainable_wts + non_trainable_wts))
# Number of training weights = 138442 and non-trainabel weights = 0.0
# Total number of parameters = 138442.0

print(f"According to tf.keras.backend: number of training weights = {trainable_wts},"
      f" non-trainable weights = {non_trainable_wts} and total number of parameters = "
      f"{trainable_wts + non_trainable_wts}")
print(f"According to tf.math.count_nonzero() method, total number of trainable weights = {tot_params}")
# According to tf.keras.backend: number of training weights = 138442, non-trainable weights = 0.0 and total number of parameters = 138442.0
# According to tf.math.count_nonzero() method, total number of trainable weights = 137760

