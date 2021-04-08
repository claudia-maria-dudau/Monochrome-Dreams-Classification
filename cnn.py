import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.constraints import maxnorm
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
import pandas as pd
import read_data as rd
import normalization as norm

train_images, train_labels, validation_images, validation_labels, test_images, test_str = rd.read_data()
scaled_train, scaled_validation, scaled_test = norm.normalization(train_images, validation_images, test_images, "minmax")
train_labels, validation_labels = norm.one_hot_encoding(train_labels, validation_labels)

# 2D -> 4D pentru cnn
scaled_train = scaled_train.reshape(30001, 32, 32, 1)
scaled_validation = scaled_validation.reshape(5000, 32, 32, 1)
scaled_test = scaled_test.reshape(5000, 32, 32, 1)

# Keras
class KerasNN:
    def __init__(self, input_shape, no_classes):
        # setarea seed-ului pentru a putea reproduce rezultatele
        np.random.seed(7)
        tf.random.set_seed(7)

        # sequential model
        self.model = Sequential()

        # layer 1 - convolution + relu
        self.model.add(Conv2D(filters=32,
                              kernel_size=(5, 5),
                              padding='same',
                              activation='relu',
                              input_shape=input_shape))

        # layer 2 - average pooling
        self.model.add(AveragePooling2D(pool_size=(2, 2),
                                        padding='same'))

        # layer 3 - convolution + relu
        self.model.add(Conv2D(filters=32,
                              kernel_size=(5, 5),
                              padding='same',
                              activation='relu'))

        # layer 4 - flattening the data
        self.model.add(Flatten())

        # layer 5 - dropout
        self.model.add(Dropout(rate=0.4))

        # layer 6 - dense + relu
        self.model.add(Dense(units=600,
                             activation='relu',
                             kernel_constraint=maxnorm(3)))

        # layer 7 - dropout
        self.model.add(Dropout(rate=0.5))

        # layer 8 - dense + softmax
        self.model.add(Dense(units=no_classes,
                             activation='softmax',
                             kernel_constraint=maxnorm(3)))

    def train(self, train_data, train_labels, validation_data, validation_labels):
        # antrenare model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='nadam',
                           metrics=['accuracy'])

        self.model.summary()
        self.history = self.model.fit(train_data, train_labels,
                                      validation_data=(validation_data, validation_labels),
                                      epochs=15,
                                      batch_size=64)

    def evaluate(self, validation_data, validation_labels):
        # elavuare model
        scores = self.model.evaluate(validation_data, validation_labels, verbose=0)
        print("Accuracy: " + str((scores[1] * 100)))

    def predict(self, test_data):
        # determinare predictii
        predictions = self.model.predict(test_data)
        labels = [0] * predictions.shape[0]
        for i in range(predictions.shape[0]):
            labels[i] = np.argmax(predictions[i])
        return labels

    def confusion_matrix(self, validation_data, validation_labels):
        # calculare matricea de confuzie
        predictions = self.predict(validation_data)
        conf_matrix = metrics.confusion_matrix(np.argmax(validation_labels, axis=1), predictions)

        # plotting
        df_cm = pd.DataFrame(conf_matrix,
                             index=[i for i in range(9)],
                             columns=[i for i in range(9)])
        sn.set(font_scale=1.2) # label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 9}) # font size
        plt.title("Confusion Matrix")
        plt.show()

    def accuracy_plotting(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

    def loss_plotting(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()


print('init')
KNN = KerasNN(scaled_train.shape[1:], train_labels.shape[1])

print('train')
KNN.train(scaled_train, train_labels, scaled_validation, validation_labels)

print('evaluate')
KNN.evaluate(scaled_validation, validation_labels)

print('predict')
test_labels = KNN.predict(scaled_test)

df = pd.DataFrame({'id': test_str,
                   'label': test_labels})
df.to_csv('predictions.csv', index=False)

# plottings
KNN.confusion_matrix(scaled_validation, validation_labels)
KNN.accuracy_plotting()
KNN.loss_plotting()