import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from itertools import cycle
from numpy import trapz

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def plot_pr(line_dfs, labels, title, filename):
    lw = 2
    plt.figure()
    for i, df in enumerate(line_dfs):
        #auc = trapz(df['precision'], df['recall'])
        plt.plot(df['recall'], df['precision'],
                 label=labels[i],# + ', AUC = ' + str(round(auc, 3)),
                 color=COLORS[i], linewidth=lw)

    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    ppn = round(line_dfs[0]['precision'][len(line_dfs[0]['precision'])-1], 3)
    plt.axhline(y=ppn, color='gray', linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(filename)
    plt.show()


def plot_roc(line_dfs, labels, title, filename):
    lw = 2
    plt.figure()
    for i, df in enumerate(line_dfs):
        auc = trapz(df['recall'], df['fpr'])
        plt.plot(df['fpr'], df['recall'],
                 label=labels[i]+', AUC = '+str(round(auc, 3)),
                 color=COLORS[i], linewidth=lw)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('books_read.png')
    plt.savefig(filename)
    plt.show()


DIR = '/Users/wang/Documents/git/lets_learn/infclean/data/tsv/eval/pdb_emb/rent_curves/'
EMB_PREFIX = DIR + 'emb_'
PLAIN_PREFIX = DIR + 'plain_'
NR_PREFIX = 'NR_'
PR_SUFFIX = 'pr_curve.csv'
ROC_SUFFIX = 'roc_curve.csv'

emb_pr_df = pd.read_csv(EMB_PREFIX+PR_SUFFIX)
emb_roc_df = pd.read_csv(EMB_PREFIX+ROC_SUFFIX)
plain_pr_df = pd.read_csv(PLAIN_PREFIX+PR_SUFFIX)
plain_roc_df = pd.read_csv(PLAIN_PREFIX+ROC_SUFFIX)
emb_nr_pr_df = pd.read_csv(EMB_PREFIX+NR_PREFIX+PR_SUFFIX)
emb_nr_roc_df = pd.read_csv(EMB_PREFIX+NR_PREFIX+ROC_SUFFIX)
plain_nr_pr_df = pd.read_csv(PLAIN_PREFIX+NR_PREFIX+PR_SUFFIX)
plain_nr_roc_df = pd.read_csv(PLAIN_PREFIX+NR_PREFIX+ROC_SUFFIX)


plot_roc([emb_roc_df, plain_roc_df], ['With embeddings', 'Without embeddings'], 'With vs. without embeddings with RA dataset - ROC Curves', 'ra_emb_roc.png')
plot_pr([emb_pr_df, plain_pr_df], ['With embeddings', 'Without embeddings'], 'With vs. without embeddings with RA dataset - PR Curves', 'ra_emb_pr.png')
plot_roc([emb_roc_df, emb_nr_roc_df], ['With ICs', 'Without ICs'], 'With vs. without ICs with RA dataset - With embeddings  - ROC Curves', 'ra_emb_ic_roc.png')
plot_pr([emb_pr_df, emb_nr_pr_df], ['With ICs', 'Without ICs'], 'With vs. without ICs with RA dataset - With embeddings  - PR Curves', 'ra_emb_ic_pr.png')
plot_roc([plain_roc_df, plain_nr_roc_df], ['With ICs', 'Without ICs'], 'With vs. without ICs with RA dataset - No embeddings - ROC Curves', 'ra_plain_ic_roc.png')
plot_pr([plain_pr_df, plain_nr_pr_df], ['With ICs', 'Without ICs'], 'With vs. without ICs with RA dataset - No embeddings - PR Curves', 'ra_plain_ic_pr.png')


