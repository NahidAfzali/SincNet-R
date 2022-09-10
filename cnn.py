import utils
import tensorflow as tf
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten

from custom_callback import CustomCallback

# Variables
cnn_model_path = "models/cnn_model"
#=============================================

# check if model already exists
try:
    model = tf.keras.models.load_model(cnn_model_path)
except:
    # (x_train, y_train) = utils.readAllEEGSignals()
    z = [None]
    utils.readEEGSignals(1, 1, z, 0)
    (x_train, y_train) = z[0]

    model = Sequential()

    number_of_outputs = 3

    model.add(Conv1D(80, kernel_size=124, activation='relu', input_shape=(2480, 1)))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(60, kernel_size=5, activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(60, kernel_size=5, activation='relu'))
    model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(number_of_outputs, activation='softmax'))

    #opt = RMSprop(learning_rate=0.001, rho=0.95, epsilon=1e-8)
    opt = Adam(learning_rate = 0.001)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
 
    model.fit(x_train, y_train, 
                epochs=400, 
                batch_size=128,
                callbacks=[CustomCallback()])

    model.save(cnn_model_path)

input('Press Enter to close the program...')