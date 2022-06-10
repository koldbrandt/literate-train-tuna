import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def get_device():
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_training_stats(stats):
    # Plot the training loss and accuracy
    plt.style.use('ggplot')
    plt.figure()
    N = np.arange(0, len(stats['train_loss']))
    plt.plot(N, stats['train_loss'], label='Training Loss')
    plt.plot(N, stats['test_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/loss.png')

    plt.close()

    plt.figure()
    plt.plot(N, stats['train_acc'], label='Training Accuracy')
    plt.plot(N, stats['test_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('figures/accuracy.png')


def save_training_stats(stats, filename):
    # Save the training statistics to a csv
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv("trainstats/"+filename)



def nms(boxes, threshold):
    """
    Non-maximum suppression.
    :param boxes: np.array of bounding boxes (x1, y1, x2, y2)
    :param threshold: overlap threshold
    :return: list of indices of bounding boxes to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(areas)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep
