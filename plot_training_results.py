import pathlib
import pickle
from training.analysis.model_analysis import plot_loss_curves
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)

    args = parser.parse_args()
    path = args.path

    with open(path, 'rb') as handle:
        model_results = pickle.load(handle)

    plot_loss_curves(model_results)
