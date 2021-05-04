""" Image processing script

Helper script to resize and/or change color mode of images.
"""


if __name__ == "__main__":

    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from tqdm import tqdm
    import glob, os, sys
    from utils import *

    help_str = f"""Usage:
-g, --help       print help message. Including this command will make the program exit.
-i, --indir      specify input directory (mandatory)
-o, --outdir     output directory (if blank, processed images will be stored in a folder in the same location as the script, with the same name as input, suffixed with '-p')
-f, --format     specify the format of outputs (useful for converting e.g., from .jpg to .png)
-w, --width      specify the width of the outputs (useful for resizing). Will be kept the same if not supplied.
-h, --height     specify the height of the outputs (useful for resizing). Will be kept the same if not supplied.
-m, --mode       specify the color mode of the outputs, e.g., RGB, RGBA, CYMK, etc.
-c, --changename whether to change the names of the files or not (yes, no)"""

    arguments, values = return_args(
        help_args,
        indir_args,
        outformat_args,
        outwidth_args,
        outheight_args,
        outmode_args,
        changename_args,
        outdir_args,
    )

    for current_argument, current_value in arguments:
        if current_argument in make_args(help_args):
            print(help_str)
            sys.exit()
        elif current_argument in make_args(indir_args):
            indir = current_value
        elif current_argument in make_args(outformat_args):
            outformat = current_value
        elif current_argument in make_args(outwidth_args):
            outwidth = int(current_value)
        elif current_argument in make_args(outheight_args):
            outheight = int(current_value)
        elif current_argument in make_args(outmode_args):
            outmode = current_value
        elif current_argument in make_args(changename_args):
            changename = current_value
        elif current_argument in make_args(outdir_args):
            outdir = current_value

    if not indir:
        print("You must speficy a directory with only images to process.")
        sys.exit(1)
    if len(arguments) == 1:
        print("You need some commands. Try changing the width or the height.")
        sys.exit(1)

    newdir = os.path.basename(indir) + "-p" if not outdir else outdir
    count = 0
    if not os.path.exists(newdir):
        os.mkdir(newdir)

    img_paths = glob.glob(indir + "/**/*.*", recursive=True)
    for img_path in tqdm(img_paths):
        pathname, extension = os.path.splitext(img_path)
        basename = os.path.basename(pathname)
        try:
            img = Image.open(img_path)
        except Exception as e:
            # print(e) # skips all files that is not an image
            continue

        inwidth, inheight = img.size
        if (not outwidth) and outheight:
            out_size = (inwidth, outheight)
        elif outwidth and (not outheight):
            out_size = (outwidth, inheight)
        elif (not outwidth) and (not outheight):
            out_size = None
        else:
            out_size = (outwidth, outheight)

        img = convert_img(img, out_size, outmode)
        outformat = outformat if outformat else extension
        if outformat == ".jpg" and img.mode in ("P", "RGBA"):
            outformat = ".png"
        count += 1
        if changename and changename == "yes":
            basename = count
        path = os.path.join(newdir, f"{basename}{outformat}")
        img.save(path)
    print(f"Processed {count} images. Files were saved in ./{newdir}")
