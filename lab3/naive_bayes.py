import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt', 'int')
test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', 'int')


def values_to_bins(arr, bins):
    return np.digitize(arr,bins)-1


def confuzion_matrix(y_true,y_pred):
    num_classes=max(y_pred.max(),y_true.max())+1
    conf_matrix=np.zeros((num_classes,num_classes))
    for i in range(len(y_true)):
        conf_matrix[int(y_true[i]),int(y_pred[i])]+=1
    return conf_matrix


def test_accuracy(i):
    bins = np.linspace(start=0, stop=255, num=i)
    x_train=values_to_bins(train_images,bins)
    x_test= values_to_bins(test_images,bins)
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(x_train, train_labels)
    return naive_bayes_model.score(x_test, test_labels)


def misclassified_examples(i):
    bins = np.linspace(start=0, stop=255, num=11)
    x_train=values_to_bins(train_images,bins)
    x_test= values_to_bins(test_images,bins)
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(x_train, train_labels)
    predicted_lables=naive_bayes_model.predict(x_test)
    missclasified=np.where(predicted_lables!=test_labels)[0]
    for j in range(i):
        image=train_images[missclasified[j],:]
        image=np.reshape(image,(28,28))
        plt.imshow(image.astype(np.uint8),cmap='gray')
        plt.title("This image was classified as "+str(predicted_lables[missclasified[i]]))
        plt.show()


print(test_accuracy(11))

misclassified_examples(5)

bins = np.linspace(start=0, stop=255, num=11)
naive_bayes_model = MultinomialNB()

x_train = values_to_bins(train_images,bins)
x_test = values_to_bins(test_images,bins)

naive_bayes_model.fit(x_train, train_labels)
predicted_lables = naive_bayes_model.predict(x_test)
print(confuzion_matrix(test_labels, predicted_lables))