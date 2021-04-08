from sklearn import preprocessing
from keras.utils import np_utils

def define_scalar(train_data, type="standard"):
    # normalizarea datelor
    if type == "standard":
        scalar = preprocessing.StandardScaler()
        scalar.fit(train_data)
    elif type == "l1":
        scalar = preprocessing.Normalizer("l1")
        scalar.fit(train_data)
    elif type == "l2":
        scalar = preprocessing.Normalizer("l2")
        scalar.fit(train_data)
    elif type == "minmax":
        scalar = preprocessing.MinMaxScaler()
        scalar.fit(train_data)
    return scalar

def normalization(train_images, validation_images, test_images, type="standard"):
    # ---------- PREPROCESARE ----------
    print("preprocesare")

    scalar = define_scalar(train_images, type)
    scaled_train = scalar.transform(train_images)
    scaled_validation = scalar.transform(validation_images)
    scaled_test = scalar.transform(test_images)

    '''
    scaled_train = train_images
    scaled_validation = validation_images
    scaled_test = test_images

    scaled_train = (train_images - np.mean(train_images, axis=0)) / np.std(train_images)
    scaled_validation = (validation_images - np.mean(validation_images, axis=0)) / np.std(validation_images)
    scaled_test = (test_images - np.mean(test_images, axis=0)) / np.std(test_images)
    '''

    return scaled_train, scaled_validation, scaled_test

def one_hot_encoding(train_labels, validation_labels):
    # one hot encoding
    train_labels = np_utils.to_categorical(train_labels)
    validation_labels = np_utils.to_categorical(validation_labels)

    return train_labels, validation_labels
