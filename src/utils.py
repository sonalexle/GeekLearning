import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import sys, getopt, os, glob
import hiddenlayer as hl

help_args = ("g", "help")
indir, indir_args = None, ("i:", "indir=")
outformat, outformat_args = None, ("f:", "format=")
outwidth, outwidth_args = None, ("w:", "width=")
outheight, outheight_args = None, ("h:", "height=")
outmode, outmode_args = None, ("m:", "mode=")
changename, changename_args = None, ("c:", "changename=")
outdir, outdir_args = None, ("o:", "outdir=")


def argparser(*args):
    parser = argparse.ArgumentParser()
    for arg in args:
        parser.add_argument(arg)
    args = parser.parse_args()
    return args


def make_options(*args):
    short_opts = ""
    long_opts = []
    for short_opt, long_opt in args:
        short_opts += short_opt
        long_opts.append(long_opt)
    return short_opts, long_opts


def make_args(arg_tuple):
    """
    Assumes that the input tuple has the correct form.
    """
    import re

    short_pattern = r"(\w)(:?)$"
    long_pattern = r"([\w_]+)(=?)$"
    first = re.match(short_pattern, arg_tuple[0]).group(1)
    second = re.match(long_pattern, arg_tuple[1]).group(1)
    return ("-" + first, "--" + second)


def return_args(*args):
    short_opts, long_opts = make_options(*args)
    if len(sys.argv) == 1:
        print("You must specify some commands. Use -g or --help for help.")
        sys.exit(1)
    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_opts, long_opts)
    except getopt.error as e:
        print(e)
        sys.exit(1)
    return arguments, values


def plot_images(images, n_rows=1):
    fig, axs = plt.subplots(n_rows, images.size(0) // n_rows, figsize=(8, 4))
    for ax, img in zip(axs.flat, images):
        img = img.permute(1, 2, 0) * 0.5 + 0.5
        ax.matshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
    plt.tight_layout(w_pad=0)


def compute_accuracy(model, *loaders, device="cuda:0"):
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for loader in loaders:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss += F.cross_entropy(outputs, labels, reduction="sum").item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    return correct / total, loss / total


def train_val_test_split(X, y, val_size, test_size=None, rs_train=None, rs_test=69):
    """
    If test_size is None: returns a train set and a validation set
    Else: returns a train set, validation set, and test set
    rs_train, rs_test: random states for train split, test split
    If test_size is not None: the test set will be produced first
    then another split is performed on the remaining data to create
    a train set and a validation set. By this way, we can produce
    the same test set but random train and validation sets if only
    rs_test is defined.
    """
    if test_size:
        val_size = test_size / (1 - test_size)
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs_test, shuffle=True, stratify=y
        )
        test = np.vstack((X_test, y_test))

    X, X_val, y, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=rs_train, shuffle=True
    )

    train = np.vstack((X, y))
    val = np.vstack((X_val, y_val))
    test_dict = {"test": test} if test_size else {}
    ret = {"train": train, "val": val}
    ret.update(test_dict)

    return ret


def test_predictions(loader, classes, *models, num_images=8, device="cuda:0"):
    with torch.no_grad():
        rand_idx = torch.randperm(loader.batch_size)[:num_images]
        images, labels = iter(loader).next()
        images = images[rand_idx]
        labels = labels[rand_idx]
        ncol = 4
        n_rows = images.size(0) // ncol
        plot_images(images, n_rows=n_rows)
        images = images.to(device)
        labels = labels.to(device)
        count = 0
        for model in models:
            model = model.to(device)
            # Compute predictions
            y = model(images)
            if count != 0:
                print()
            if count == 0:
                count += 1
            print(f"{model.__class__.__name__}, (Ground truth, prediction):")
            for i in range(num_images):
                ground_truth = classes[labels[i]]
                prediction = classes[y[i].argmax()]
                print((ground_truth, prediction), end=", ")
                if i % ncol == ncol - 1:
                    print()


def prepare_csv(h5file, path, write=True):
    df = {}
    n = 0
    for label in list(h5file.keys()):
        classgroup = h5file[label]
        for imgname in list(classgroup.keys()):
            df.update({n: {"imagepath": imgname, "class": label}})
            n += 1
    df = pd.DataFrame.from_dict(df, orient="index", columns=["imagepath", "class"])
    if write:
        df.to_csv(path)
    return df


def read_from_csv(csvpath, df=None):
    if not df:
        df = pd.read_csv(csvpath)
    paths = df["imagepath"].to_numpy()
    labels = df["class"].to_numpy()
    return np.vstack((paths, labels))


def make_df(indir, outdir=None, csv=False, header=True):

    classname = os.path.basename(indir)
    result = {}
    i = 0
    for filepath in glob.iglob(indir + "/**/*.*", recursive=True):
        result.update({i: [filepath, classname]})
        i += 1
    result = pd.DataFrame.from_dict(
        result, orient="index", columns=["imagepath", "class"]
    )
    if csv:
        result.to_csv(outdir, mode="a", header=header, index=False)
    return result


def countfiles(path):
    count = 0
    for filepath in glob.iglob(path + "/**/*.*", recursive=True):
        count += 1
    return count


def render_graph(
    model,
    save_path,
    transforms=None,
    format="pdf",
    orient="LR",
    sample_tensor=torch.zeros([1, 3, 64, 64])
):
    if transforms:
        graph = hl.build_graph(model, sample_tensor, transforms=transforms)
    else:
        graph = hl.build_graph(model, sample_tensor)
    dot = graph.build_dot()
    dot.attr("graph", rankdir=orient) #Left-Right
    dot.format = format
    dot.render(save_path)


if __name__ == "__main__":
    import shutil, glob, sys, os
    from tqdm import tqdm

    help_str = (
        "-g for help, -i for input directories, -o for name and location of the csv"
    )
    arguments, values = return_args(help_args, indir_args, outdir_args)
    for current_argument, current_value in arguments:
        if current_argument in make_args(help_args):
            print(help_str)
            sys.exit()
        elif current_argument in make_args(indir_args):
            indir = current_value
        elif current_argument in make_args(outdir_args):
            outdir = current_value
    if not indir:
        print("You need at least one input directory")
        sys.exit(1)
    outdir = outdir if outdir else "imagesandlabels.csv"
    make_df(indir, outdir, csv=True, header=not os.path.exists(outdir))
