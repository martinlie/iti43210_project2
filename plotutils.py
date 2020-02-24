import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OneHotEncoder

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = np.array(classes)[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    #fix for bug in jupyter
    ax.set_ylim(len(classes) - .5, - .5)

    return ax

def read_data():

    class_names = ["Tumor",  "Stroma",  "Complex", "Lympho", "Debris",  "Mucosa",  "Adipose",  "Empty"]

    mcd = pd.read_csv('../colorectal-histology-mnist/' + 'hmnist_64_64_L.csv')
    # Split the data set into independent(x) and dependent (y) data sets where the last column is the y
    x_len = mcd.shape[1]-1
    x = mcd.iloc[:,0:x_len].values
    y = mcd.iloc[:,x_len].values
    
    # Normalized histogram extraction method
    x_hist = []
    for xe in x:
        hist, _ = np.histogram(xe, bins=16, density=False) 
        x_hist.append(hist)
    x_hist=np.array(x_hist)

    # One hot encoding
    onehot = OneHotEncoder(sparse=False)
    y_hot = onehot.fit_transform(y.reshape(len(y), 1))

    return x_hist,y_hot,onehot,class_names

def show_confusion(y_test_true, y_test_pred,class_names,mlcm=False):
    cm = confusion_matrix(y_test_true, y_test_pred)
    print(cm)

    if mlcm:
        mlcm = multilabel_confusion_matrix(y_test_true, y_test_pred)
        for j in range(len(mlcm)): 
            cm=mlcm[j]
            TP=cm[0][0]
            TN=cm[1][1]
            FN=cm[1][0]
            FP=cm[0][1]

            print("Class ", class_names[j])
            print(cm)
            c_accuracy = (TP+TN) / (TP+TN+FN+FP)
            c_error = (FP+FN) / (TP+TN+FN+FP)
            c_sensitivity = TP / (TP + FN)
            c_specificity = TN / (FP + TN)
            print("Accuracy = ", c_accuracy)
            print("Error rate = ", c_error)
            print("Sensitivity = ", c_sensitivity)
            print("Specificity = ", c_specificity)

    plot_confusion_matrix(y_test_true-1, y_test_pred-1, classes=class_names, title='Confusion matrix')
    plt.show()