# A script for making HDF5 files from image folders
# Expected structure: see https://pytorch.org/vision/stable/datasets.html#imagefolder
# Note that inside the root directory (and all sub-dirs) there SHOULD NOT be any non-image files.

if __name__ == "__main__":
    import h5py, os, glob, argparse
    import numpy as np
    from tqdm import tqdm
    from PIL import Image

    # ONLY WORKS WHEN SUPPLIED ABSOLUTE PATHS
    parser = argparse.ArgumentParser()
    parser.add_argument("basepath", help="Specify the folder containing the data")
    parser.add_argument("savepath", help="Specify path to save hdf5 file")
    args = parser.parse_args()

    base_path = args.basepath  # dataset path
    save_path = args.savepath  # path to save the hdf5 file

    hf = h5py.File(save_path, "w")  # open the file in append mode

    for classname in os.listdir(base_path):
        classpath = os.path.join(base_path, classname)
        grp = hf.create_group(classname)
        img_paths = glob.glob(classpath + "/**/*.*", recursive=True)
        for img_path in tqdm(img_paths):
            with open(img_path, "rb") as img_f:
                binary_data = img_f.read()
            binary_data_np = np.asarray(binary_data)
            dset = grp.create_dataset(img_path, data=binary_data_np)

    hf.close()
