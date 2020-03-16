from __future__ import print_function
import keras
import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, LSTM, Embedding
from keras.models import Model
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
max_features = 20000
batch_size = 8
num_classes = 3
epochs = 17
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_surface_roughness_trained_model.h5'

# The data, split between train and test sets:
x_train, x_test, x_train_1, x_test_1, x_train_2, x_test_2,\
y_train, y_test = train_test_split(np.load('train_data_YCbCr_Histogram.npy'),
                                   np.load('train_data_YCbCr_stnd_dev.npy'),
                                   np.load('train_data_YCbCr_mean.npy'),
                                   np.load('train_label_YCbCr_Histogram.npy'),
                                   test_size=0.1, random_state=123)

# print('x_train shape:', x_train.shape)
print('y_test shape:', y_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

stnd_dev = Input(shape=(1,))
mean = Input(shape=(1,))

LSTM_layer = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
End_LSTM_Layer = LSTM(128, dropout=0.5, recurrent_dropout=0.5)

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model1_input = Input(shape=x_train.shape[1:])

print(model1_input.shape)

model1 = Conv2D(32, (3, 3), padding='same',
                data_format='channels_first')(model1_input)
model1 = Activation('relu')(model1)
model1 = Conv2D(32, (3, 3))(model1)
model1 = Activation('relu')(model1)
model1 = MaxPooling2D(pool_size=(2, 2))(model1)
model1 = Dropout(0.25)(model1)

model1 = Conv2D(64, (3, 3), padding='same')(model1)
model1 = Activation('relu')(model1)
model1 = Conv2D(64, (3, 3))(model1)
model1 = Activation('relu')(model1)
model1 = MaxPooling2D(pool_size=(2, 2))(model1)
model1 = Dropout(0.25)(model1)

model1 = Conv2D(128, (3, 3), padding='same')(model1)
model1 = Activation('relu')(model1)
model1 = Conv2D(128, (3, 3))(model1)
model1 = Activation('relu')(model1)
model1 = MaxPooling2D(pool_size=(2, 2))(model1)
model1 = Dropout(0.25)(model1)

model1 = Flatten()(model1)
model1 = Dense(512)(model1)
model1 = Activation('relu')(model1)
model1 = Dropout(0.5)(model1)

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

embedded_s = Embedding(max_features, 128)(stnd_dev)
embedded_m = Embedding(max_features, 128)(mean)
encoded_s_1 = LSTM_layer(embedded_s)
encoded_m_1 = LSTM_layer(embedded_m)
encoded_s_2 = LSTM_layer(encoded_s_1)
encoded_m_2 = LSTM_layer(encoded_m_1)
encoded_s_3 = LSTM_layer(encoded_s_2)
encoded_m_3 = LSTM_layer(encoded_m_2)
encoded_s_4 = LSTM_layer(encoded_s_3)
encoded_m_4 = LSTM_layer(encoded_m_3)
encoded_s_5 = LSTM_layer(encoded_s_4)
encoded_m_5 = LSTM_layer(encoded_m_4)
encoded_s_6 = End_LSTM_Layer(encoded_s_5)
encoded_m_6 = End_LSTM_Layer(encoded_m_5)

print('Build model...')

merged_vector = keras.layers.concatenate([encoded_s_6, encoded_m_6, model1])

predictions = Dense(3, activation='softmax')(merged_vector)

model = Model(inputs=[stnd_dev, mean, model1_input], outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


#y_train = y_train.reshape(244, 3)
#y_test = y_test.reshape(61, 3)

history = model.fit([x_train_1, x_train_2, x_train], y_train, batch_size, epochs,
                    validation_data=([x_test_1, x_test_2, x_test], y_test))

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

score, acc = model.evaluate([x_test_1, x_test_2, x_test], y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()