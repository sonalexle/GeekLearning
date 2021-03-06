# A script containing various utilities for the project

import pandas as pd
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import sys, getopt, os, glob
import hiddenlayer as hl
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# These are for command line argument parsing in some other scripts.
# This parsing solution was implemented before I knew the existence of the `argparse` library.
# And I'm too lazy to re-implement CL arg parsing in those scripts.
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


def compute_accuracy(
    model,
    *loaders,
    device="cuda:0",
    accuracy=True,
    criterion=torch.nn.CrossEntropyLoss()
):
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    acc, total, loss = 0, 0, 0
    # To calculate average loss over all batches:
    # running_loss += batch_loss * size_of_current_batch (reduction='mean')
    # Divide running_loss by total of samples: sum_i len(loader_i.sampler)
    # or accumulate sizes of each batch in a loop.
    # Another way: append batch losses in a list (reduction='mean')
    # and take the mean of that list.
    # That is, divide the sum of batch losses by the number of batches.
    # Never use reduction='sum' (?) because some sums are not over the batches
    # e.g., multi-target NLL-loss.
    with torch.no_grad():
        for loader in loaders:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                predicted = torch.argmax(outputs, dim=1)
                acc += (predicted == labels).sum().item()
    acc = acc/total
    if not accuracy:
        acc = 1 - acc # 0/1 loss = 1 - accuracy
    return acc, loss/total


def train_val_test_split(X, y, val_size, test_size=None, rs_train=None, rs_test=69):
    """
    Args:
        test_size: optional proportion value. If not supplied,
            returns a train set and a validation set, 
            else returns a train set, validation set, and test set
        rs_train, rs_test: random states for train split, test split
    If test_size is supplied, the test set will be produced first
    then another split is performed on the remaining data to create
    a train set and a validation set. Using this method, we can produce
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
        images, labels = images[rand_idx], labels[rand_idx]
        ncol = 4
        n_rows = images.size(0) // ncol
        plot_images(images, n_rows=n_rows)
        images, labels = images.to(device), labels.to(device)
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


def read_from_csv(csvpath=None, df=None):
    assert csvpath is not None or df is not None, "at least one arg should be supplied"
    if df is None:
        df = pd.read_csv(csvpath)
    df = df.astype({"class": 'int64'})
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


def convert_img(img, out_size=None, out_mode=None):
    if out_size:
        img = img.resize(out_size)
    if out_mode:
        img = img.convert(out_mode)
    return img


def predict_one(img_path, model, classes, input_size=(64, 64), input_mode="RGB"):
    pathname, extension = os.path.splitext(img_path)
    basename = os.path.basename(pathname)
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(e)
        return
    img = convert_img(img, out_size=input_size, out_mode=input_mode)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    img = transform(img).unsqueeze(0)
    pred = model(img)
    return img, pred


def to_torchscript_and_save(model, filename, method):
    assert method == 'script' or method == 'trace', "method should be \'script\' or \'trace\'"
    script = model.to_torchscript(method=method)
    torch.jit.freeze(script).save(filename)

