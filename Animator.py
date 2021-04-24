from images2gif import writeGif
from PIL import Image
import imageio
import os


# def make_gif(folderpath=None, filename=None, ftype=".png", duration=1):
    
#     file_names = sorted((fn for fn in os.listdir(folderpath) if fn.endswith(ftype)))
#     for i in range(len(file_names)):
#         file_names[i] = folderpath + "/" + file_names[i]
        
#     # try:
#         images = [Image.open(fn) for fn in file_names]
#          writeGif(filename, images, duration=duration)
#     # except:
#         print("Cannot make gif, check if folderpath is empty, address is valid or "
#            + "library is installed properly")
#     return


def make_gif(path=None, fname="t.gif", duration=1):
    
    image_folder = os.fsencode(path)
    filenames = []
    
    for file in os.listdir(image_folder):
        filename = os.fsdecode(file)
        if filename.endswith( ('.jpeg', '.png', '.gif') ):
            filenames.append(path + filename)
    
    # filenames.sort()
    images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join(fname), images, duration=duration) 
    
    return