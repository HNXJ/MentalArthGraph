from images2gif import writeGif
from PIL import Image
import imageio
import pickle
import os


def save_list(l, filename="List0.txt"):
        
    with open(filename, "wb") as f_temp:
        pickle.dump(l, f_temp)
    
    print("Saved.")
    return


def load_list(filename="List0.txt"):
    
    with open(filename, "rb") as f_temp:
        l = pickle.load(f_temp)
    
    print("Loaded.")
    return l


def make_gif(path=None, fname="t.gif", duration=1, f=75):
    
    image_folder = os.fsencode(path)
    filenames = []
    cnt = 0
    
    for file in os.listdir(image_folder):
        cnt += 1
        if cnt > f:
            continue
        filename = os.fsdecode(file)
        if filename.endswith( ('.jpeg', '.png', '.gif') ):
            filenames.append(path + filename)
    
    # filenames.sort()
    images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join(fname), images, duration=duration) 
    print("Gif created successfully.")
    return


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

