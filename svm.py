from sklearn import svm
from sklearn import metrics
import read_data as rd
import normalization as norm

train_images, train_labels, validation_images, validation_labels, test_images, test_str = rd.read_data()
scaled_train, scaled_validation, scaled_test = norm.normalization(train_images, validation_images, test_images)

# Support Vector Machine
class SupportVectorMachine:
    def __init__(self, c, type="linear"):
        self.SVC = svm.SVC(C=c, kernel=type)

    def train(self, train_data, train_labels):
        # antrenare model
        self.SVC.fit(train_data, train_labels)

    def test(self, test_data, test_labels):
        # calculare predictii
        predictions = self.SVC.predict(test_data)
        return metrics.accuracy_score(test_labels, predictions) * 100


max_svm = (0, 0)
for c in range(-10, 10):
    print("init")
    SVM = SupportVectorMachine(10 ** c)

    print("train")
    SVM.train(scaled_train, train_labels)

    print("predict")
    accuracy = SVM.test(scaled_validation, validation_labels)

    print(accuracy, c)
    if accuracy > max_svm[0]:
        max_svm = (accuracy, c)
print(max_svm)