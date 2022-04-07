import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from itertools import cycle
import pandas as pd
import tensorflow as tf


def get_figure_repr_1(X_train):
    fig, axs = plt.subplots(1, 4, sharey="row", figsize=(10, 4))

    plt.rcParams["text.usetex"] = True

    plt.tight_layout()
    plt.style.use("seaborn-notebook")

    cmap = "viridis"
    im_0 = X_train[X_train["Chern_bulk"] == 0].to_numpy()[0][0].reshape(24, 24)
    axs[0].imshow(im_0, cmap=cmap)
    axs[0].set_title("$|C| = 0$")

    im_1 = X_train[X_train["Chern_bulk"] == 1].to_numpy()[15][0].reshape(24, 24)
    pcm = axs[1].imshow(im_1, cmap=cmap)
    axs[1].set_title("$|C| = 1$")

    im_2 = X_train[X_train["Chern_bulk"] == 2].to_numpy()[15][0].reshape(24, 24)
    axs[2].imshow(im_2, cmap=cmap)
    axs[2].set_title("$|C| = 2$")

    im_3 = X_train[X_train["Chern_bulk"] == 3].to_numpy()[16][0].reshape(24, 24)
    axs[3].imshow(im_3, cmap=cmap)
    axs[3].set_title("$|C| = 3$")

    fig.text(0.5, 0.25, "X pixels", ha="center")
    fig.text(0.005, 0.6, "Y pixels", va="center", rotation="vertical")
    fig.colorbar(pcm, ax=[axs], location="bottom", label="LDOS Colormap")

    for ax in axs.flat:
        ax.set(frame_on=False, xticks=np.arange(0, 24, 5), yticks=np.arange(0, 24, 5))

    for ax in axs.flat:
        ax.label_outer()


# Code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def roc_auc(model, X, Y, title=None):
    if title is None:
        title = "A multiclass ROC characteristic"
    y_score = model.predict(X)
    y_test = Y.numpy()
    n_classes = Y[0].shape[0]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = plt.figure(figsize=(10, 5))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    return fig


def plotHistScores(model, test_set, test_labels):
    data = getScores(model, test_set, test_labels)
    proc_0 = data[0].to_numpy()[0] / data[0].to_numpy().sum() * 100
    print(proc_0)

    proc_1 = data[1].to_numpy()[1] / data[1].to_numpy().sum() * 100
    print(proc_1)

    proc_2 = data[2].to_numpy()[2] / data[2].to_numpy().sum() * 100
    print(proc_2)

    proc_3 = data[3].to_numpy()[3] / data[3].to_numpy().sum() * 100
    print(proc_3)

    fig, ax = plt.subplots()

    ax.bar(
        [0, 1, 2, 3],
        [proc_0, proc_1, proc_2, proc_2],
        width=0.1,
        tick_label=["Cat.0", "Cat.1", "Cat.2", "Cat.3"],
        color=["cyan", "red", "green", "blue"],
    )
    ax.set_title("Procent poprawnie sklasyfikowanych obrazków")
    ax.grid()
    ax.set_yticks(np.arange(0, 100, 10))

    fig.tight_layout()
    plt.show()


def getScores(model, X, Y):
    y_pred = model.predict(X)
    y_true = Y

    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    matrix = confusion_matrix(y_true, y_pred)

    return pd.DataFrame(matrix)


def printScores(model, X, Y):
    y_pred = model.predict(X)
    y_true = Y

    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


def plotTrainingHistory(model):

    history = model.history
    epochs_num = len(history["accuracy"])
    epoch_range = np.linspace(1, epochs_num, num=epochs_num)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(epoch_range, history["loss"], color="cyan")
    axes[0].plot(epoch_range, history["val_loss"], color="orange")
    axes[0].set_ylabel("Loss function value")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(["train", "validation"], loc="upper right")

    axes[1].plot(epoch_range, history["accuracy"], color="cyan")
    axes[1].plot(epoch_range, history["val_accuracy"], color="orange")
    axes[1].set_ylabel("Accuracy score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(["train", "validation"], loc="lower right")

    fig.show()
