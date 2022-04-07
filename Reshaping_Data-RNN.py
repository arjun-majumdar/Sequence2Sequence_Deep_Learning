

import numpy as np
import pandas as pd


"""
Data reshaping for Multivariate multi-step time series prediction/forecasting.

There is a need to reshape a given dataset because RNN/GRU/LSTM Sequence Modeling using Deep
Learning expect the input data to be shaped as a 3-D tensor or multi-dimensional array-
[batch_size, look_back, num_features]

1. look_back is the number of past values or sequence you use for each training step/iteration.

2. num_features is the number of features/attributes/channels/variables/columns to be used for
multivariate forecasting.
"""


# Remember: During slicing, the last index is ignored!!

# Create a 2-D random data sampled from a normal distribution-
x = np.random.normal(loc = 1.2, scale = 0.56, size = (10, 3))

x
'''
array([[0.59645659, 1.98797682, 1.97874871],
       [0.63471794, 1.03300396, 1.41402465],
       [0.32856824, 1.26653781, 1.23833057],
       [0.96555612, 0.75536962, 1.9164309 ],
       [1.58389118, 1.29623228, 1.22239347],
       [1.03936399, 1.52845841, 0.90619389],
       [1.68795405, 0.55613747, 1.40958416],
       [1.05087044, 0.33980013, 1.39977023],
       [0.95728804, 1.06490829, 0.78810827],
       [1.49555889, 1.34954422, 1.2938286 ]])
'''

# NOTE: This sample dataset has the assumption that the target attribute is in the last column.
# However, going ahead, this can easily be changed for other sitatuation(s)!


# Specify number of past values to be used for training-
look_back = 5

# Specify number of future values to be predicted-
future_window = 2

# Python3 lists to contain training data and target-
X, y = list(), list()


# Loop through the 2-D matrix dataset to create 'X' and 'y'-
i = look_back

# Assumption: target/y is the last column in 'x', if not, change code accordingly.

while i <= (len(x) - future_window):
    
    X.append(x[i - look_back: i, :3])
    
    # This does NOT include the current target value for 'i' but rather
    # i + 1 and i + 2-
    # x[i: i + future_window, 2]
    y.append(x[i: i + future_window, 2])
    
    # This INCLUDES the current target value for 'i', as in-
    # i and i + 1-
    # x[i - 1: i + future_window - 1, 2]
    
    i += 1


# Convert from list to np arrays-
X = np.asarray(X)
y = np.asarray(y)

# Sanity check-
X.shape, y.shape
# ((4, 5, 3), (4, 2))


# Further sanity check-
X
'''
array([[[0.59645659, 1.98797682, 1.97874871],
        [0.63471794, 1.03300396, 1.41402465],
        [0.32856824, 1.26653781, 1.23833057],
        [0.96555612, 0.75536962, 1.9164309 ],
        [1.58389118, 1.29623228, 1.22239347]],

       [[0.63471794, 1.03300396, 1.41402465],
        [0.32856824, 1.26653781, 1.23833057],
        [0.96555612, 0.75536962, 1.9164309 ],
        [1.58389118, 1.29623228, 1.22239347],
        [1.03936399, 1.52845841, 0.90619389]],

       [[0.32856824, 1.26653781, 1.23833057],
        [0.96555612, 0.75536962, 1.9164309 ],
        [1.58389118, 1.29623228, 1.22239347],
        [1.03936399, 1.52845841, 0.90619389],
        [1.68795405, 0.55613747, 1.40958416]],

       [[0.96555612, 0.75536962, 1.9164309 ],
        [1.58389118, 1.29623228, 1.22239347],
        [1.03936399, 1.52845841, 0.90619389],
        [1.68795405, 0.55613747, 1.40958416],
        [1.05087044, 0.33980013, 1.39977023]]])
'''

y
'''
array([[0.90619389, 1.40958416],
       [1.40958416, 1.39977023],
       [1.39977023, 0.78810827],
       [0.78810827, 1.2938286 ]])
'''


# To show a more practical example using a sample dataset-
data = pd.read_csv("sample_data.csv")

data.dtypes
'''
date    object
col1     int64
col2     int64
dtype: object
'''

data.head()
'''
       date  col1  col2
0  1-Apr-22    84   217
1  2-Apr-22    72   138
2  3-Apr-22    28   164
3  4-Apr-22     6   267
4  5-Apr-22     1   110
'''


def prepare_data_rnn(x, look_back = 5, future_window = 2):
    '''
    Function to input 2-D data and convert it into a 3-D tensor
    required by deep learning architectures such as - RNN, LSTM, GRU, etc.
    
    Assumption: target/y is the last column in 'x'. If otherwise, change the code.
    
    Inputs:
    1. x - numpy ndarray
    2. look_back - number of past data points/samples to use for training
    3. future_window - number of future values to predict
    
    Returns:
    1. X - 3-D tensor
    2. y - accompanying target
    '''
    # Python3 lists to contain training data and target-
    X, y = list(), list()

    # Loop through the 2-D matrix dataset to create 'X' and 'y'-
    i = look_back

    while i <= (len(x) - future_window):
    
        X.append(x[i - look_back: i, :3])
    
        # This does NOT include the current target value for 'i' but rather
        # i + 1 and i + 2-
        # x[i: i + future_window, 2]
        y.append(x[i: i + future_window, 2])
    
        # This INCLUDES the current target value for 'i', as in-
        # i and i + 1-
        # x[i - 1: i + future_window - 1, 2]
    
        i += 1


    # Convert from list to np arrays-
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y


X, y = prepare_data_rnn(x = data.values, look_back = 5, future_window = 2)

X.shape, y.shape
# ((24, 5, 3), (24, 2))

# Use the first 5 rows (from 01-Apr-22 until 05-Apr-22) as input-
X[0, :]
'''
array([['1-Apr-22', 84, 217],
       ['2-Apr-22', 72, 138],
       ['3-Apr-22', 28, 164],
       ['4-Apr-22', 6, 267],
       ['5-Apr-22', 1, 110]], dtype=object)
'''

# To predict the next two target values-
y[0]
# array([153, 116], dtype=object)


# Two data points using 5 rows each according to 'look_back' parameter-
X[:2, :]
'''
array([[['1-Apr-22', 84, 217],
        ['2-Apr-22', 72, 138],
        ['3-Apr-22', 28, 164],
        ['4-Apr-22', 6, 267],
        ['5-Apr-22', 1, 110]],

       [['2-Apr-22', 72, 138],
        ['3-Apr-22', 28, 164],
        ['4-Apr-22', 6, 267],
        ['5-Apr-22', 1, 110],
        ['6-Apr-22', 99, 153]]], dtype=object)
'''

# To predict two 2 steps in the future according to 'future_window' parameter.
# This is due to multivariate multi-step time series forecasting problem.
y[:2]
'''
array([[153, 116],
       [116, 255]], dtype=object)
'''

# NOTE: each sample/data point has the shape (1, 5, 3) or, (5, 3) and has an
# associated target value of (2,) due to multivariate multi-step time series forecasting
# problem formulation.

