import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from itertools import cycle
import pandas as pd
import itertools
import seaborn as sns
import tensorflow as tf


def get_figure_repr_1(X_train):
    fig, axs = plt.subplots(1, 4, sharey="row", figsize=(10, 4))
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
def roc_auc(model, X, Y, title=None, labels=None):

    if labels is None:
        labels = [0, 1, 2, 3]
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
    for i, color in zip(range(len(labels)), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(labels[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    # Plot probability distributions
    fig_hist = plt.figure(figsize=(12, 8))
    fig_hist.suptitle("Classification figure")
    classes = range(len(labels))
    for i in range(len(classes)):
        # Gets the class
        c = classes[i]

        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame(X.numpy().reshape(-1, 576))
        df_aux['class'] = tf.argmax(y_test, axis=1)
        df_aux['prob'] = y_score[:, i]
        df_aux = df_aux.reset_index(drop = True)

        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 2, i+1)
        sns.histplot(x = "prob", data = df_aux, hue = 'class', ax = ax, stat="count", discrete=True, palette="viridis", legend=True)
        ax.set_title(labels[c])
        ax.legend([f"Class: {labels[3]}", f"Class: {labels[2]}", f"Class: {labels[1]}", f"Class: {labels[0]}"])
        ax.set_xlabel(f"P({labels[c]})")
    fig_hist.subplots_adjust(hspace=0.5)

    return fig, fig_hist


def plotHistScores(
    model,
    test_set,
    test_labels,
    title="Correct Classification Percentage",
    tick_labels=None,
):
    if tick_labels is None:
        tick_labels = ["Cat.0", "Cat.1", "Cat.2", "Cat.3"]
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
        tick_label=tick_labels,
        color=["cyan", "red", "green", "blue"],
    )
    ax.set_title(title)
    ax.grid()
    ax.set_yticks(np.arange(0, 100, 10))

    fig.tight_layout()
    plt.show()


def plotHistScores_tab(
    model_tab,
    test_set,
    test_labels,
    title="Correct Classification Percentage",
    tick_labels=None,
):
    if tick_labels is None:
        tick_labels = ["Cat.0", "Cat.1", "Cat.2", "Cat.3"]
    scale = len(model_tab)
    fig = plt.figure(figsize=(scale * 7, 10))
    plt.suptitle(title)
    for num, model in enumerate(model_tab):
        data = getScores(model, test_set, test_labels)
        proc_0 = data[0].to_numpy()[0] / data[0].to_numpy().sum() * 100
        # print(proc_0)

        proc_1 = data[1].to_numpy()[1] / data[1].to_numpy().sum() * 100
        # print(proc_1)

        proc_2 = data[2].to_numpy()[2] / data[2].to_numpy().sum() * 100
        # print(proc_2)

        proc_3 = data[3].to_numpy()[3] / data[3].to_numpy().sum() * 100
        # print(proc_3)

        ax = fig.add_subplot(1, len(model_tab), num + 1)

        ax.bar(
            [0, 1, 2, 3],
            [proc_0, proc_1, proc_2, proc_2],
            width=0.1,
            tick_label=tick_labels,
            color=["cyan", "red", "green", "blue"],
        )
        ax.set_title(f"Run no.{num}")
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


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


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

    # Uncomment when not using in Jupyter Notebook
    # fig.show()


def plotTrainingHistory_avg(model_tab, alpha=0.5):
    avg_accuracy = []
    avg_loss = []
    avg_val_accuracy = []
    avg_val_loss = []

    for model in model_tab:
        history = model.history

        avg_accuracy.append(np.asarray(history["accuracy"]))
        avg_loss.append(np.asarray(history["loss"]))
        avg_val_accuracy.append(np.asarray(history["val_accuracy"]))
        avg_val_loss.append(np.asarray(history["val_loss"]))

    avg_accuracy = np.asarray(avg_accuracy)
    avg_loss = np.asarray(avg_loss)
    avg_val_accuracy = np.asarray(avg_val_accuracy)
    avg_val_loss = np.asarray(avg_val_loss)

    avg_accuracy_std = avg_accuracy.std(axis=0)
    avg_loss_std = avg_loss.std(axis=0)
    avg_val_accuracy_std = avg_val_accuracy.std(axis=0)
    avg_val_loss_std = avg_val_loss.std(axis=0)

    avg_accuracy = avg_accuracy.mean(axis=0)
    avg_loss = avg_loss.mean(axis=0)
    avg_val_accuracy = avg_val_accuracy.mean(axis=0)
    avg_val_loss = avg_val_loss.mean(axis=0)

    epochs_num = len(avg_accuracy)
    epoch_range = np.linspace(1, epochs_num, num=epochs_num)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(epoch_range, avg_loss, color="cyan", label="train")
    axes[0].fill_between(
        epoch_range,
        np.add(avg_loss, avg_loss_std),
        np.subtract(avg_loss, avg_loss_std),
        color="cyan",
        alpha=alpha,
        label="train \nconfidence interval",
    )
    axes[0].plot(epoch_range, avg_val_loss, color="orange", label="validation")
    axes[0].fill_between(
        epoch_range,
        np.add(avg_val_loss, avg_val_loss_std),
        np.subtract(avg_val_loss, avg_val_loss_std),
        color="orange",
        alpha=alpha,
        label="validation \nconfidence interval",
    )
    axes[0].set_ylabel("Loss function value")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(loc="upper right")

    axes[1].plot(epoch_range, avg_accuracy, color="cyan", label="train")
    axes[1].fill_between(
        epoch_range,
        np.add(avg_accuracy, avg_accuracy_std),
        np.subtract(avg_accuracy, avg_accuracy_std),
        color="cyan",
        alpha=alpha,
        label="train \nconfidence interval",
    )
    axes[1].plot(epoch_range, avg_val_accuracy, color="orange", label="validation")
    axes[1].fill_between(
        epoch_range,
        np.add(avg_val_accuracy, avg_val_accuracy_std),
        np.subtract(avg_val_accuracy, avg_val_accuracy_std),
        color="orange",
        alpha=alpha,
        label="validation\nconfidence interval",
    )
    axes[1].set_ylabel("Accuracy score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="lower right")

    # Uncomment when not using in Jupyter Notebook
    # fig.show()
