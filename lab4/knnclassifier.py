import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt', 'int')
test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', 'int')


class KnnClassifier:

    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        if (metric == 'l2'):
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))
        elif (metric == 'l1'):
            distances = np.sum(abs(self.train_images - test_image), axis=1)
        else:
            print('Error! Metric {} is not defined!'.format(metric))

        sort_index = np.argsort(distances)
        sort_index = sort_index[:num_neighbors]
        nearest_labels = self.train_labels[sort_index]
        histc = np.bincount(nearest_labels)

        return np.argmax(histc)

    def classify_images(self, test_images, num_neighbors=3, metric='l2'):
        num_test_images = test_images.shape[0]
        predicted_labels = np.zeros((num_test_images))

        for i in range(num_test_images):
            if (i % 50 == 0):
                print('processed {}%'.format((i / num_test_images) * 100))
            predicted_labels[i] = self.classify_image(test_images[i, :], num_neighbors=num_neighbors, metric=metric)

        return predicted_labels


clf = KnnClassifier(train_images, train_labels)

test_neighbors = [1, 3, 5, 7, 9]

acc_l1 = []
acc_l2 = []

for i in test_neighbors:
    print("\nl1 with {} neighbors:".format(i))
    predicted_labels = clf.classify_images(test_images, i, 'l1')
    acc = accuracy_score(predicted_labels, test_labels)
    acc_l1.append(acc)

for i in test_neighbors:
    print("\nl2 with {} neighbors:".format(i))
    predicted_labels = clf.classify_images(test_images, i)
    acc = accuracy_score(predicted_labels, test_labels)
    acc_l2.append(acc)

text_acc_l1 = []
text_acc_l2 = []
for i in range(5):
    text_acc_l1.append('accuracy for {} neighbors: {}'.format(test_neighbors[i], acc_l1[i]))
    text_acc_l2.append('accuracy for {} neighbors: {}'.format(test_neighbors[i], acc_l2[i]))
np.savetxt('accuracies/accuracies_l1.txt', text_acc_l1, fmt="%s")
np.savetxt('accuracies/accuracies_l2.txt', text_acc_l2, fmt="%s")

plt.plot(test_neighbors, acc_l1)
plt.plot(test_neighbors, acc_l2)
plt.xlabel('number of neighbors')
plt.ylabel('accuracy')
plt.show()