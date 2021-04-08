import numpy as np
import imageio

def read_data():
    # ---------- TRAIN ----------
    print("citire train")

    # citire imagini
    train_images = np.zeros((30001, 32, 32), 'int')
    for i in range(30001):
        if i < 10:
            train_images[i] = imageio.imread(f'./data/train/00000{i}.png')
        elif i < 100:
            train_images[i] = imageio.imread(f'./data/train/0000{i}.png')
        elif i < 1000:
            train_images[i] = imageio.imread(f'./data/train/000{i}.png')
        elif i < 10000:
            train_images[i] = imageio.imread(f'./data/train/00{i}.png')
        else:
            train_images[i] = imageio.imread(f'./data/train/0{i}.png')

    # 3-D -> 2-D (fiecare imagine este reprzentata pe un rand in loc de o matrice)
    train_images = train_images.reshape(30001, 1024)

    # citire etichete
    f = open('./data/train.txt')
    train_labels = np.zeros(30001, 'int')
    for i in range(30001):
        train_labels[i] = int(f.readline().split(',')[1])
    f.close()

    # ---------- VALIDATION ----------
    print("citire validation")

    # citire imagini
    validation_images = np.zeros((5000, 32, 32), 'int')
    validation_str = []
    for i in range(30001, 35001):
        validation_images[i - 30001] = imageio.imread(f'./data/validation/0{i}.png')
        validation_str.append('0' + str(i) + '.png')

    # 3-D -> 2-D (fiecare imagine este reprzentata pe un rand in loc de o matrice)
    validation_images = validation_images.reshape(5000, 1024)

    # citire etichete
    f = open('./data/validation.txt')
    validation_labels = np.zeros(5000, 'int')
    for i in range(30001, 35001):
        validation_labels[i - 30001] = int(f.readline().split(',')[1])
    f.close()

    # ---------- TEST ----------
    print("citire test")

    # citire imagini
    test_images = np.zeros((5000, 32, 32), 'int')
    test_str = []
    for i in range(35001, 40001):
        test_images[i - 35001] = imageio.imread(f'./data/test/0{i}.png')
        test_str.append('0' + str(i) + '.png')

    # 3-D -> 2-D (fiecare imagine este reprzentata pe un rand in loc de o matrice)
    test_images = test_images.reshape(5000, 1024)

    return train_images, train_labels, validation_images, validation_labels, test_images, test_str