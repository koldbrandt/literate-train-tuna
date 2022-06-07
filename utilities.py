import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
<<<<<<< HEAD
=======
import torch


>>>>>>> 9103dc199abb5a3a43b308f5bbd3c2ea137fbc13
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