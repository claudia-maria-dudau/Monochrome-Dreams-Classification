from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np
import read_data as rd

train_images, train_labels, validation_images, validation_labels, test_images, test_str = rd.read_data()

# Naive Bayes
class NaiveBayes:
    def __init__(self):
        self.naive_bayes = MultinomialNB()

    def values_to_bins(self, data, bins):
        # pentru fiecare pixel se returneaza intervalul din care face parte
        return np.digitize(data, bins) - 1

    def train(self, train_data, no_bins):
        # preprocesare date - tranformarea in intervale
        bins = np.linspace(start=0, stop=255, num=no_bins)
        train = self.values_to_bins(train_data, bins)

        # antrenare model
        self.naive_bayes.fit(train, train_labels)

    def test(self, test_data, test_labels, no_bins):
        # preprocesare date - tranformarea in intervale
        bins = np.linspace(start=0, stop=255, num=no_bins)
        test = self.values_to_bins(test_data, bins)

        # calculare predictii
        self.naive_bayes.predict(test)
        return self.naive_bayes.score(test, test_labels) * 100


NB = NaiveBayes()
max_nb = (0, 0)
bins = []
acc = []

for no_bins in range(1, 50, 2):
    print(no_bins)
    NB.train(train_images, no_bins)
    accuracy = NB.test(validation_images, validation_labels, no_bins)
    bins.append(no_bins)
    acc.append(accuracy)
    if accuracy > max_nb[0]:
        max_nb = (accuracy, no_bins)
print(max_nb)

#plotting
plt.plot(bins, acc, '*')
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('no_bins')
plt.show()