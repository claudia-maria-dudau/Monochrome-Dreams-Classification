import numpy as np
import read_data as rd
import normalization as norm

train_images, train_labels, validation_images, validation_labels, test_images, test_str = rd.read_data()
scaled_train, scaled_validation, scaled_test = norm.normalization(train_images, validation_images, test_images)

# Knn classifier
class KnnClassifier:
    def __init__(self, train_data, train_labels):
        self.train_data = train_images
        self.train_labels = train_labels

    def l1(self, elem, data):
        # calculare dist l1
        return np.sum(np.abs(data - elem), axis=1)

    def l2(self, elem, data):
        # calculare dist l2
        return np.sqrt(np.sum((data - elem) ** 2, axis=1))

    def classify_elem(self, elem, no_neighbors=3, metric='l2'):
        # determinare distanta fata de setul de date de antrenare pentru un anumit element
        if metric == "l1":
            dist = self.l1(elem, self.train_data)
        else:
            dist = self.l2(elem, self.train_data)

        # determinare cei mai apropriati vecini
        nearest_neighbors = np.argsort(dist)[:no_neighbors]
        nearest_labels = self.train_labels[nearest_neighbors]
        prediction = np.argmax(np.bincount(nearest_labels))
        return prediction

    def classify_data(self, test_data, no_neighbors, metric):
        # determinare predictii
        predictions = np.zeros(test_data.shape[0])
        for i in range(len(test_data)):
            predictions[i] = self.classify_elem(test_data[i], no_neighbors, metric)
        return (predictions == validation_labels).mean() * 100


KNN = KnnClassifier(scaled_train, train_labels)
max_knn = (0, 0)
for no_neighbors in range(1, 50, 2):
    accuracy = KNN.classify_data(scaled_validation, no_neighbors, "l1")
    print(no_neighbors)

    if accuracy > max_knn[0]:
        max_knn = (accuracy, no_neighbors)
print(max_knn)