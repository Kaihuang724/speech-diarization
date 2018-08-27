import subprocess
from PIL import Image
import glob

image_list = []

def record_learn(name):
    arg = name
    subprocess.call("./learn.sh '%s'" % str(arg), shell=True)
    folder_name = "../data/train/{}/*.png".format(name)
    for filename in glob.glob(folder_name):
        image_list.append(filename)
    return(image_list)
