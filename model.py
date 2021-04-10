
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam


def GRUModel(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, 15, strides=4)(X_input)              # CONV1D
    X = BatchNormalization() (X)                         # Batch normalization
    X = Activation('relu') (X)                           # ReLu activation
    X = Dropout(0.8) (X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True, reset_after=False)(X)     # GRU (use 128 units and return the sequences)
    X = Dropout(0.8) (X)                                 # dropout (use 0.8)
    X = BatchNormalization() (X)                         # Batch normalization
    
    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True, reset_after=False)(X)     # GRU (use 128 units and return the sequences)
    X = Dropout(0.8) (X)                                 # dropout (use 0.8)
    X = BatchNormalization() (X)                         # Batch normalization
    X = Dropout(0.8) (X)                                 # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)
    
    return model