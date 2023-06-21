import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score
from itertools import cycle
import pandas as pd
import itertools
import seaborn as sns
import tensorflow as tf
from matplotlib import ticker


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
        ax.grid(visible=False)

    for ax in axs.flat:
        ax.label_outer()


def get_figure_repr_1_comparison_disorder(X_cleaned, X_disroder, num=10):
    fig, axs = plt.subplots(2, 4, sharey="row", figsize=(8, 4))
    plt.tight_layout()
    plt.style.use("seaborn-notebook")

    cmap = "viridis"

    for row in range(2):
        for idx in range(4):
            if row == 0:
                cm_pic = X_cleaned.loc[X_cleaned["Chern_bulk"] == idx].to_numpy()[num][0].reshape(24, 24)
                # cm_pic = X_cleaned[cm_loc]
                pcm = axs[row, idx].imshow(cm_pic, cmap=cmap)
                axs[row, idx].set_title(f"$|C| = {idx}$")
            else:
                cm_pic = X_disroder.loc[X_disroder["Chern_bulk"] == idx].to_numpy()[num][0].reshape(24, 24)
                # cm_pic = X_disroder[cm_loc].numpy().reshape(24, 24)
                pcm = axs[row, idx].imshow(cm_pic, cmap=cmap)
                axs[row, idx].set_title(f"$|C| = {idx}$")

    fig.text(0.5, 0.25, "X pixels", ha="center")
    fig.text(0.005, 0.6, "Y pixels", va="center", rotation="vertical")
    fig.text(0.04, 0.85, "$V = 0$", va="center", rotation="vertical")
    fig.text(0.04, 0.45, "$V > 1$", va="center", rotation="vertical")
    fig.colorbar(pcm, ax=[axs], location="bottom", label="LDOS Colormap")

    for ax in axs.flat:
        ax.set(frame_on=False, xticks=np.arange(0, 24, 5), yticks=np.arange(0, 24, 5))
        ax.grid(visible=False)

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
            label="ROC curve of class {0} (area = {1:0.2f})".format(
                labels[i], roc_auc[i]
            ),
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
        df_aux["class"] = tf.argmax(y_test, axis=1)
        df_aux["prob"] = y_score[:, i]
        df_aux = df_aux.reset_index(drop=True)

        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 2, i + 1)
        sns.histplot(
            x="prob",
            data=df_aux,
            hue="class",
            ax=ax,
            stat="count",
            discrete=True,
            palette="viridis",
            legend=True,
        )
        ax.set_title(labels[c])
        ax.legend(
            [
                f"Class: {labels[3]}",
                f"Class: {labels[2]}",
                f"Class: {labels[1]}",
                f"Class: {labels[0]}",
            ]
        )
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
    # print(proc_0)

    proc_1 = data[1].to_numpy()[1] / data[1].to_numpy().sum() * 100
    # print(proc_1)

    proc_2 = data[2].to_numpy()[2] / data[2].to_numpy().sum() * 100
    # print(proc_2)

    proc_3 = data[3].to_numpy()[3] / data[3].to_numpy().sum() * 100
    # print(proc_3)

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


def printScores(model, X, Y, get_acc=False, supress=False):
    """
    :param model: Model you want to test
    :param X: Data
    :param Y: Values
    :param get_acc: Should the function return accuracy score
    :param supress: Supresses stdout
    :return: Default: Confusion Matrix, if get_acc == True: tuple(Confusion Matrix, Accuracy)
    """
    y_pred = model.predict(X)
    y_true = Y

    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    if not supress:
        print(cr)
    if not get_acc:
        return cm
    else:
        return cm, acc


def plot_confusion_matrix(cm, class_names, title=None, cmap="coolwarm", percent=True, supress_labels=False):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
       percent: Should the function plot percentile of class memebers on each tile
       cmap: Colormap to use
       title: Title of the plot
       supress_labels: Should superimposed labels be dropped
    """
    if title is None and not supress_labels:
        title = "Confusion matrix"
    figure = plt.figure(figsize=(8, 8))
    im = plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.grid(visible=False)
    if not supress_labels:
        plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0

    if not supress_labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, "{:.2f}%".format(cm[i, j]*100), horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.close("all")
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


def plotTrainingHistory_avg(model_tab, alpha=0.5, figshape=(10, 5), plot_loss=True, title=None):
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

    if plot_loss:
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
    else:
        fig, axes = plt.subplots(1, 1, figsize=figshape)
        axes.plot(epoch_range, avg_accuracy, color="cyan", label="train")
        axes.fill_between(
            epoch_range,
            np.add(avg_accuracy, avg_accuracy_std),
            np.subtract(avg_accuracy, avg_accuracy_std),
            color="cyan",
            alpha=alpha,
            label="train \nconfidence interval",
        )
        axes.plot(epoch_range, avg_val_accuracy, color="orange", label="validation")
        axes.fill_between(
            epoch_range,
            np.add(avg_val_accuracy, avg_val_accuracy_std),
            np.subtract(avg_val_accuracy, avg_val_accuracy_std),
            color="orange",
            alpha=alpha,
            label="validation\nconfidence interval",
        )
        axes.set_ylabel("Accuracy score")
        axes.set_xlabel("Epoch")
        axes.legend(loc="lower right")
        fig.suptitle(title)


def plotTrainingHistory_avg_model_tab(model_tab_group, alpha=0.5, figshape=(10, 5), title_tab=None, save_title=None):
    plt.close("All")
    model_tab_data = []
    if title_tab is None:
        title_tab = []
        for num in range(len(model_tab_group)):
            title_tab.append("Model No.{}".format(num))
    for model_tab in model_tab_group:
        avg_accuracy = []
        avg_val_accuracy = []

        for model in model_tab:
            history = model.history

            avg_accuracy.append(np.asarray(history["accuracy"]))
            avg_val_accuracy.append(np.asarray(history["val_accuracy"]))

        avg_accuracy = np.asarray(avg_accuracy)
        avg_val_accuracy = np.asarray(avg_val_accuracy)

        avg_accuracy_std = avg_accuracy.std(axis=0)
        avg_val_accuracy_std = avg_val_accuracy.std(axis=0)

        avg_accuracy = avg_accuracy.mean(axis=0)
        avg_val_accuracy = avg_val_accuracy.mean(axis=0)

        epochs_num = len(avg_accuracy)
        epoch_range = np.linspace(1, epochs_num, num=epochs_num)
        model_tab_data.append([epoch_range, avg_accuracy, avg_accuracy_std, avg_val_accuracy, avg_val_accuracy_std])

    fig, axes = plt.subplots(1, 3, figsize=figshape)
    # plt.style.use("seaborn-notebook")
    for num, data in enumerate(model_tab_data):
        title = title_tab[num]
        epoch_range, avg_accuracy, avg_accuracy_std, avg_val_accuracy, avg_val_accuracy_std = data
        axes[num].plot(epoch_range, avg_accuracy, color="cornflowerblue", label="Training Dataset")
        axes[num].fill_between(
            epoch_range,
            np.add(avg_accuracy, avg_accuracy_std),
            np.subtract(avg_accuracy, avg_accuracy_std),
            color="cornflowerblue",
            alpha=alpha,
            label="_train \nconfidence interval",
        )
        axes[num].plot(epoch_range, avg_val_accuracy, color="orange", label="Validation Dataset")
        axes[num].fill_between(
            epoch_range,
            np.add(avg_val_accuracy, avg_val_accuracy_std),
            np.subtract(avg_val_accuracy, avg_val_accuracy_std),
            color="orange",
            alpha=alpha,
            label="_validation\nconfidence interval",
        )
        axes[num].set_ylabel("Accuracy score")
        axes[num].set_xlabel("Epoch")
        axes[num].set_ylim(0.92, 1.0)
        axes[num].legend(loc="lower right")
        axes[num].set_title(title)
        # axes[num].grid(False)
        # axes[num].set_frame_on(False)
        axes[num].axis("on")
        formatter = ticker.PercentFormatter(xmax=1.0)
        axes[num].xaxis.set_major_locator(ticker.MultipleLocator(20))
        axes[num].yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        axes[num].yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
        axes[num].yaxis.set_major_formatter(formatter)
        axes[num].yaxis.set_minor_formatter(ticker.NullFormatter())

    for ax in axes.flat:
        ax.label_outer()

    for spine in plt.gca().spines.values():
        spine.set_visible(True)

    if save_title is not None:
        plt.savefig(save_title)


    # Uncomment when not using in Jupyter Notebook
    # fig.show()
