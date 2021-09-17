import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
import logging
log = logging.getLogger(__name__)
plt.style.use('dark_background')


def save_results(train_losses, train_accs, test_losses, test_accs, results_path):
    df = pd.DataFrame({  # save model training process into csv file
        'loss': train_losses,
        'test_loss': test_losses,
        'acc': train_accs,
        'test_acc': test_accs
    })
    df.to_csv(results_path)


def plot_losses_accs(dry_run, results_path):
    df = pd.read_csv(results_path)
    test_accs = df['test_acc']
    train_accs = df['acc']
    test_losses = df['test_loss']
    train_losses = df['loss']
    log.info(f'best train accuracy: {train_accs.max()*100:.2f} at epoch {train_accs.idxmax()+1}')
    log.info(f'best test accuracy: {test_accs.max()*100:.4f} at epoch {test_accs.idxmax()+1}')
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    ax[0].plot(train_losses, color='red')
    ax[0].plot(test_losses, color='green')
    ax[0].set_xlabel('Epoch', size=16)
    ax[0].set_ylabel('Loss', size=16)
    ax[0].grid(alpha=0.5)
    ax[0].tick_params(labelsize=16)
    ax[0].legend(['Train', 'Test'], loc='right', fontsize=16)
    if dry_run:
        ax[0].set_xticks([0, 1, 2, 3, 4])
        ax[0].set_ylim(0.035, 0.050)
        ax[0].set_yticks([0.035, 0.04, 0.045, 0.050])
    else:
        ax[0].set_xticks([0, 100, 200, 300, 400])
        ax[0].set_ylim(0.9, 1.1)
        ax[0].set_yticks([0.9, 1.0, 1.1, 1.2])
    
    ax[1].plot(train_accs, color='red')
    ax[1].plot(test_accs, color='green')
    ax[1].set_xlabel('Epoch', size=16)
    ax[1].set_ylabel('Accuracy', size=16, labelpad=-5)
    ax[1].grid(alpha=0.5)
    ax[1].tick_params(labelsize=16)
    ax[1].legend(['Train', 'Test'], loc='right', fontsize=16)
    yt = ax[1].get_yticks()
    ax[1].set_yticklabels(['{:,.0%}'.format(x) for x in yt])
    if dry_run:
        ax[1].set_xticks([0, 1, 2, 3, 4])
        ax[1].set_ylim(0.2, 0.5)
        ax[1].set_yticks([0.2, 0.3, 0.4, 0.5])
    else:
        ax[1].set_xticks([0, 100, 200, 300, 400])
        ax[1].set_ylim(0.7, 1.0)
        ax[1].set_yticks([0.7, 0.8, 0.9, 1.0])

    fig.savefig('loss_acc_conv2_split.png', bbox_inches='tight')