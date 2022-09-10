from keras.callbacks import Callback
import numpy as np

class CustomCallback(Callback):
    def on_train_begin(self, logs=None):
        self.accuracies = []
        self.epochs = []

    def on_train_end(self, logs=None):
        with open('plot_data/sincnet.npy', 'wb') as f:
            np.save(f, np.array(self.accuracies))
            np.save(f, np.array(self.epochs))

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        self.accuracies.append(accuracy)
        self.epochs.append(epoch)
        if epoch == 200:
            self.model.optimizer.learning_rate = 0.0005