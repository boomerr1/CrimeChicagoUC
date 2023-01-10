import tensorflow as tf
from tensorflow.keras import Input, layers, Model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

################################ Creating Mask #################################
mask = np.ones((55,50))
mask = tf.keras.backend.constant(mask)
mask = tf.expand_dims(mask, -1)
################################################################################

lookback = 7
batch_size = 32

# Create data with shape (3653, 55, 50, 1) with 3653 timesteps
data = np.random.random((3653, 55, 50, 1))

train_gen = TimeseriesGenerator(
    data,
    data,
    length=lookback,
    batch_size=batch_size,
    shuffle=False
)

def masked_MSE_loss(y_true, y_pred):
    y_pred_masked = tf.math.multiply(y_pred, mask)
    mse = tf.keras.losses.mean_squared_error(y_true = y_true, y_pred = y_pred_masked)
    return mse

# Define the input tensors
inputs = Input(shape=(lookback, 55, 50, 1))

# First stack of convlstm layers
convlstm1 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=True)(inputs)
bathnorm1 = layers.BatchNormalization()(convlstm1)
convlstm2 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=False)(bathnorm1)

# Second stack of convlstm layers
convlstm3 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=True)(inputs)
batchnorm2 = layers.BatchNormalization()(convlstm3)
convlstm4 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=False)(batchnorm2)

# Concatenate outputs of two stacks
concatenation = layers.concatenate([convlstm2, convlstm4])
outputs = layers.Conv2D(filters=1, kernel_size=1, padding="same", activation='tanh')(concatenation)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss=masked_MSE_loss, metrics=['mae'])

for batch in train_gen:
    batch_input, batch_target = batch
    model.fit(x=batch_input, y=batch_target, epochs=100)