from images2gif import writeGif
from PIL import Image
import os


def make_gif(folderpath=None, filename=None, ftype=".png", duration=1):
    
    file_names = sorted((fn for fn in os.listdir('.') if fn.endswith(ftype)))
    images = [Image.open(fn) for fn in file_names]
    writeGif(filename, images, duration=duration)
    return

