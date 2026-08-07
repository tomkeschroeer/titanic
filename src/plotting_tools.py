import matplotlib.pyplot as plt
import os
import numpy as np

def plot_preds_against_truth(preds, truth, outpath):
    if preds.shape != truth.shape:
        raise ValueError("predictions and truth values must have the same shape")

    correct = sum(preds == truth)/len(preds)
    incorrect = 1 - correct

    plt.figure()
    plt.bar(["Correct", "Incorrect"], [correct, incorrect], color=["green", "red"],label=[f"correct = {round(correct,4)}", f"incorrect = {round(incorrect,4)}"])
    plt.ylabel("Count")
    plt.title("Prediction Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outpath}/plots/corr_incorr.pdf")
    plt.clf()

def plot_survived_not_survived(preds, truth, outpath):
    if preds.shape != truth.shape:
        raise ValueError("predictions and truth values must have the same shape")
    
    preds_surv = sum(preds == 1)
    preds_died = len(preds) - preds_surv

    truth_surv = sum(truth == 1)
    truth_died = len(truth) - truth_surv

    plt.figure()
    plt.title("Number of Survivers and Deceased")
    plt.ylabel("Fraction of passengers")
    plt.xlabel("Label")
    plt.tight_layout()
    plt.hist(preds, color="green", fill=True, bins=[-0.5,0.5,1.5], density=True, alpha=0.5)
    plt.hist(truth, color="red", fill=True, bins=[-0.5,0.5,1.5], density=True, alpha=0.5)
    plt.savefig(f"{outpath}/plots/surv.pdf")
    plt.close()

def plot_inputs(df, labels, input_feat, outpath, postfix=None):
    discrete_vars = [
        "Sex",
        "Pclass",
        "SibSp",
        "Parch"
    ]
    breakpoint()
    if not postfix:
        input_plot_dirs = f"{outpath}/plots/inputs/".replace("//","/")
    else:
        input_plot_dirs = f"{outpath}/plots/inputs_{postfix}/".replace("//","/")
    os.makedirs(input_plot_dirs, exist_ok=True)
    for inp in input_feat:
        plt.figure()
        plt.ylabel("Number of people")
        plt.tight_layout()
        plt.xlabel(inp)
        data = df[inp]
        if inp in discrete_vars:
            bins = np.arange(min(data)-0.5, max(data)+1.5, 1)
            plt.hist(data[labels == 1], bins=bins, color="green", fill=True, density=True, alpha=0.5, label="survived")
        else:
            _, bins, _ = plt.hist(data[labels == 1], color="green", fill=True, density=True, alpha=0.5, label="survived")
        plt.hist(data[labels == 0], bins=bins, color="red", fill=True, density=True, alpha=0.5, label="deceased")
        plt.legend()
        plt.savefig(f"{input_plot_dirs}/{inp}.pdf")
        plt.close()
    plt.close()
