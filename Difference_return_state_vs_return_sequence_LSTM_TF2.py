

"""
Difference between return_sequence and return_state for LSTMs in TF2 Keras

As part of TF2 LSTM implementation, the Keras API provides access to both
'return sequences' and 'return state'. The use and difference between these
data can be confusing when designing sophisticated recurrent neural network
models, such as the encoder-decoder model.


Summary-
1. That 'return sequences' returns the hidden state output for each input time
step.

2. That return state returns the hidden state output and cell state for the last
input time step.

3. That return sequences and return state can be used at the same time.


Long Short-Term Memory RNN

- Creating a layer of LSTM memory units allows you to specify the number of memory
units within the layer.

- Each unit or cell within the layer has an 'internal cell state', often abbreviated
as 'c', and outputs a 'hidden state', often abbreviated as 'h'.

The Keras API allows you to access these data, which can be useful or even required
when developing sophisticated recurrent neural network architectures, such as the
encoder-decoder model.

For the rest of this tutorial, we will look at the API for access these data.


Reference-
https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
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
Return Sequences

Each LSTM cell will output one hidden state 'h' for each input.

We can demonstrate this in Keras with a very small model having a single
LSTM layer that contains a single LSTM cell.

Here, we will have one input sample with 3 time steps and one feature observed
at each time step as follows-

t1 = 0.1
t2 = 0.2
t3 = 0.3
"""

# Specify number of features and number of time steps-
num_features = 1
num_time_steps = 3

# Initialize an LSTM layer having one LSTM cell-
lstm_layer = LSTM(
    units = 1, input_shape = (num_time_steps, num_features)
)

# Create input data having one feature with 3 time steps. 'X'
# is reshaped as- (batch_size, num_time_steps, num_features)
X = np.array([0.1, 0.2, 0.3]).reshape((1, 3, 1))

X.shape
# (1, 3, 1)

# Get LSTM output for a single hidden state using given input
# sequence having 1 feature with 3 time steps-
output = lstm_layer(X)

output.shape
# TensorShape([1, 1])

print(f"Output for a single hidden state for an input with one feature and 3 time steps = {output.numpy()[0][0]:.4f}")
# Output for a single hidden state for an input with one feature and 3 time steps = -0.0552


# Define an LSTM layer to accept two features with 3 time steps-
num_features = 2

# Define input having 2 features with 3 time steps-
X = np.random.randn(3, 2).reshape(1, 3, 2)

X.shape
# (1, 3, 2)

# Initialize an LSTM layer having 2 LSTM cells to accept 2 features-
lstm_layer = LSTM(
    units = 2, input_shape = (num_time_steps, num_features)
)
# Remember: Each LSTM cell will output one hidden state 'h' for each input.

output = lstm_layer(X)

output.shape
# TensorShape([1, 2])

print(f"Output for 2 hidden states for an input with 2 features and 3 time steps = {output.numpy()[0]}")
# Output for 2 hidden states for an input with 2 features and 3 time steps = [ 0.17687586 -0.10709888]


# Initialize an LSTM layer having 3 LSTM cells to accept 2 features-
lstm_layer = LSTM(
    units = 3, input_shape = (num_time_steps, num_features)
)

# Input having batch-size = 5 with 3 time steps and 2 features-
X = np.random.randn(5, 3, 2)

X.shape
# (5, 3, 2)

# LSTM output has shape: (batch_size, number_LSTMcells)
output = lstm_layer(X)

output.shape
# TensorShape([5, 3])




"""
Access the 'hidden state' output for each input time step

It is possible to access the 'hidden state' output for each input
time step. This can be done by setting the 'return_sequences' attribute
to True when defining the LSTM layer, as follows-
"""

num_features = 1
num_time_steps = 3

# Define LSTM layer to access hidden state output for each input time step-
lstm_layer = LSTM(
    units = 1, return_sequences = True,
    input_shape = (num_time_steps, num_features)
)

# Define input to LSTM with batch-size = 1, 3 time steps and 1 feature-
X = np.random.randn(1, 3, 1)

X.shape
# (1, 3, 1)

output = lstm_layer(X)

output.shape
# TensorShape([1, 3, 1])

print(f"Output for each time step:\n{output.numpy()}")
'''
Output for each time step:
[[[-0.01339091]
  [ 0.22812685]
  [ 0.22888325]]]
'''


"""
Note:

1. You must set 'return_sequences = True' when stacking LSTM layers so
that the second LSTM layer has a three-dimensional sequence input. For more
details, refer to Stacked Long Short-Term Memory Networks-
https://machinelearningmastery.com/stacked-long-short-term-memory-networks/


2. You may also need to access the sequence of hidden state outputs when
predicting a sequence of outputs with a 'Dense' output layer wrapped in a
'TimeDistributed' layer. Refer to How to Use the TimeDistributed Layer for
Long Short-Term Memory Networks in Python-
https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
"""




"""
Return States

- 'The output of an LSTM cell or layer of cells is called the hidden state'.
This is confusing, because 'each LSTM cell retains an internal state that is
not output, called the cell state, or c'.

- Generally, we do not need to access the cell state unless we are developing
sophisticated models where subsequent layers may need to have their cell state
initialized with the final cell state of another layer, such as in an
encoder-decoder model.


Keras provides the 'return_state' argument in the LSTM layer that will provide
access to the hidden state output (state_h) and the cell state (state_c). For example-

lstm1, state_h, state_c = LSTM(units = 1, return_state = True)

This may look confusing because both 'lstm1' and 'state_h' refer to the same
hidden state output. The reason for these two tensors being separate will become
clear in the next section.

We can demonstrate access to the hidden and cell states of the cells in the LSTM layer
as follows-
"""

num_features = 1
num_time_steps = 3

X.shape
# (1, 3, 1)

# Define LSTM layer-
lstm_layer = LSTM(
    units = 1, return_state = True,
    input_shape = (num_time_steps, num_features)
)

lstm1, state_h, state_c = lstm_layer(X)
# NOTE: 'lstm1' and 'state_h' refer to the same hidden state output!

lstm1.shape, state_h.shape, state_c.shape
# (TensorShape([1, 1]), TensorShape([1, 1]), TensorShape([1, 1]))

lstm1.numpy(), state_h.numpy(), state_c.numpy()
'''
(array([[0.0402967]], dtype=float32),
 array([[0.0402967]], dtype=float32),
 array([[0.07033518]], dtype=float32))
'''


# Define LSTM model using Functional API-
inputs_lstm = Input(shape = (num_time_steps, num_features))

lstm1, state_h, state_c = LSTM(
    units = 1, return_state = True,
    input_shape = (num_time_steps, num_features))(inputs_lstm)

model = Model(inputs = inputs_lstm, outputs = [lstm1, state_h, state_c])

# Get LSTM's predictions-
output = model.predict(X)

output
'''
[array([[0.01717509]], dtype=float32),
 array([[0.01717509]], dtype=float32),
 array([[0.02931273]], dtype=float32)]
'''

'''
Running the example returns 3 arrays:

- The LSTM hidden state output for the last time step.
- The LSTM hidden state output for the last time step (again).
- The LSTM cell state for the last time step.

The hidden state and the cell state could in turn be used to initialize
the states of another LSTM layer with the same number of cells.
'''


# Sanity check (without using Model)-
lstm_layer = LSTM(
    units = 1, return_state = True,
    input_shape = (num_time_steps, num_features)
)

[x.numpy() for x in lstm_layer(X)]
'''
[array([[0.01936848]], dtype=float32),
 array([[0.01936848]], dtype=float32),
 array([[0.04586496]], dtype=float32)]
'''

lstm_layer(X)
'''
[<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.01936848]], dtype=float32)>,
 <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.01936848]], dtype=float32)>,
 <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.04586496]], dtype=float32)>]
'''


# Define input having batch_size = 64,
# look_back_window = 50 (time steps) and num_features = 8-
X = np.random.randn(64, 50, 8)

X.shape
# (64, 50, 8)

num_time_steps = X.shape[1]
num_features = X.shape[2]

print(f"Input data with batch-size = {X.shape[0]}, num_features = {num_features} & look-back window size = {num_time_steps}")
# Input data with batch-size = 64, num_features = 8 & look-back window size = 50

lstm_layer = LSTM(
    units = 1, return_state = True,
    input_shape = (num_time_steps, num_features)
)

lstm1, state_h, state_c = lstm_layer(X)

lstm1.shape, state_h.shape, state_c.shape
# (TensorShape([64, 1]), TensorShape([64, 1]), TensorShape([64, 1]))

# Output shape: (batch_size, number of LSTMcells)


# Sanity check- Define the same LSTM without returning output
# state for each time step-
lstm_layer = LSTM(
    units = 1, return_state = False,
    input_shape = (num_time_steps, num_features)
)

output = lstm_layer(X)

output.shape
# TensorShape([64, 1])




"""

Return States and Sequences

We can access both the sequence of hidden state and the cell states
at the same time. This can be done by configuring the LSTM layer to
return both sequences and return states. An examples is-

lstm1, state_h, state_c = LSTM(1, return_sequences = True, return_state = True)
"""
num_time_steps = 3
num_features = 1

# Define LSTM model-
inputs1 = Input(shape = (num_time_steps, num_features))
lstm1, state_h, state_c = LSTM(
    units = 1, return_sequences = True,
    return_state = True)(inputs1)

model = Model(inputs = inputs1, outputs = [lstm1, state_h, state_c])

X = np.random.randn(1, num_time_steps, num_features)

X.shape
# (1, 3, 1)

# Make predictions-
model.predict(X)
'''
[array([[[0.19090763],
         [0.16495043],
         [0.02000636]]], dtype=float32),
 array([[0.02000636]], dtype=float32),
 array([[0.04914938]], dtype=float32)]
'''

'''
This code example shows why the LSTM output tensor and hidden state output
tensor are declared separably.

- This LSTM layer returns:
    a. the hidden state output for each/all input time steps,
    b. (then separately,) the hidden state output for the last time step (and)
    c. the cell state output for the last time step

- This can be confirmed by seeing that the last value in the returned sequences
(first array) matches the value in the hidden state (second array).
'''


# Sanity check (without using Model)-
lstm_layer = LSTM(
    units = 1, return_state = True,
    return_sequences = True, input_shape = (num_time_steps, num_features)
)

X.shape
# (1, 3, 1)

lstm_layer(X)
'''
[<tf.Tensor: shape=(1, 3, 1), dtype=float32, numpy=
 array([[[0.3976129 ],
         [0.2559724 ],
         [0.15805882]]], dtype=float32)>,
 <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.15805882]], dtype=float32)>,
 <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.44910914]], dtype=float32)>]
'''


# A more practical example-
num_features = 8
num_time_steps = 50

# Define an LSTM layer having 128 LSTMcells or units-
lstm_layer = LSTM(
    units = 128, return_state = True,
    return_sequences = True, input_shape = (num_time_steps, num_features)
)

# Define input with batch-size = 64-
X = np.random.randn(64, num_time_steps, num_features)

X.shape
# (64, 50, 8)

# Get LSTM layer's outputs-
h_state_all_time_steps, h_state_op_last_time_step, c_state_op_last_time_step = lstm_layer(X)

h_state_all_time_steps.shape, h_state_op_last_time_step.shape, c_state_op_last_time_step.shape
# (TensorShape([64, 50, 128]), TensorShape([64, 128]), TensorShape([64, 128]))




"""
Summary:

- 'return_sequences' parameter returns the hidden state output for all input time
step.

- 'return_state' argument returns the hidden state output and cell state output for
the last input time step.

- 'return_sequences' and 'return_state' arguments can be used at the same time.
"""

