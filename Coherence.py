import csv
import glob
import numpy as np
from PIL import Image

from scipy import signal
from sklearn import metrics
from pyedflib import highlevel
from matplotlib import pyplot as plt

from statsmodels.tsa.stattools import grangercausalitytests as gct


def get_sensor_locs(mode="EEG"):
    sensor_locs = np.zeros([21, 2])
    sensor_locs = np.array([[0.41, 0.18], [0.59, 0.18], [0.37, 0.33], [0.63, 0.33], 
                        [0.25, 0.3], [0.75, 0.3], [0.2, 0.5], [0.8, 0.5],
                        [0.35, 0.5], [0.65, 0.5], [0.25, 0.7], [0.75, 0.7],
                        [0.37, 0.68], [0.63, 0.68], [0.4, 0.82], [0.6, 0.82], 
                        [0.5, 0.33], [0.5, 0.5], [0.5, 0.68], [0.1, 0.5], 
                        [0.5, 0.1]])
    return sensor_locs


def load_image(infilename) :
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename) :
    img = Image.fromarray(np.asarray( np.clip(npdata,0,255), dtype="uint8")
                          , "L")
    img.save(outfilename)
    return


def load_file(k=0, foldername="EEGMA/"):
        
    fields = [] 
    rows = []       
    with open(foldername + "subject-info.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader)  
        for row in csvreader: 
            rows.append(row)  
        
    #
    
    try:
        signals, signal_headers, header = highlevel.read_edf(foldername + "Subject" + str(k) + "_2.edf")
    except:
        signals, signal_headers, header = highlevel.read_edf(foldername + "Subject0" + str(k) + "_2.edf")
    print("\n", fields[4], " | ", rows[k][0], " | ", rows[k][4])
    print("######## --- > ")
    
    return signals, signal_headers, header, fields, rows


def coherence(Temps, signals, method="Pearson", channels=None, Overlap=None, 
              filter_order=4, lbf=20, ubf=32, sampling_rate=500):
    
    # chs = [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 18, 19]        
    chs = [] 
    if channels == None:
        for i in range(21):
            chs.append(i)
    else:
        chs = channels
    
    Interval_size = int(signals.shape[1]/Temps)
    cor = np.zeros([len(chs), len(chs), Temps])
        
    nyq = sampling_rate/2
    b, a = signal.butter(filter_order, [lbf/nyq, ubf/nyq], btype='band')
    for k in range(21):
        signals[k, :] = signal.lfilter(b, a, signals[k, :])
    
    for i in range(len(chs)): #signals.shape[0]):
        for j in range(len(chs)):
            for l in range(Temps):
                ie = chs[i]
                je = chs[j]
                h = signals[ie, l*Interval_size:(l+1)*Interval_size]
                x = signals[je, l*Interval_size:(l+1)*Interval_size]
                
                if method == "Phase":
                    xs = np.fft.fft(x)
                    hs = np.fft.fft(h)
                    ys = np.correlate(hs, xs)                    
                    p = np.real(ys)**2 / (np.imag(ys)**2 + np.real(ys)**2) * (np.pi/2)
                    # print(np.real(ys))
                    if np.real(ys) > 0:    
                        cor[i][j][l] = p
                    else:
                        cor[i][j][l] = np.pi + p
                
                elif method == "Spectral":
                    xs = np.abs(np.fft.fft(x))
                    hs = np.abs(np.fft.fft(h))
                    ys = np.correlate(hs - np.mean(hs), xs - np.mean(xs))                    
                    sigms = np.std(hs)*np.std(xs)*hs.shape[0]
                    cor[i][j][l] = ys/sigms
                
                else:
                    y = np.correlate(h - np.mean(h), x - np.mean(x))
                    sigm = np.std(h)*np.std(x)*h.shape[0]
                    cor[i][j][l] = y/sigm
        
    return cor            


def mutual_info(Temps, signals, method="Temporal", channels=None, Overlap=None, 
              filter_order=4, lbf=20, ubf=32, sampling_rate=500):
    
    # chs = [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 18, 19]        
    chs = [] 
    if channels == None:
        for i in range(21):
            chs.append(i)
    else:
        chs = channels
    
    Interval_size = int(signals.shape[1]/Temps)
    cor = np.zeros([len(chs), len(chs), Temps])
        
    nyq = sampling_rate/2
    b, a = signal.butter(filter_order, [lbf/nyq, ubf/nyq], btype='band')
    for k in range(21):
        signals[k, :] = signal.lfilter(b, a, signals[k, :])
    
    for i in range(len(chs)): #signals.shape[0]):
        for j in range(len(chs)):
            for l in range(Temps):
                ie = chs[i]
                je = chs[j]
                h = signals[ie, l*Interval_size:(l+1)*Interval_size]
                x = signals[je, l*Interval_size:(l+1)*Interval_size]
                
                if method == "Phase":
                    xs = np.fft.fft(x)
                    hs = np.fft.fft(h)
                    ys = metrics.mutual_info_score(hs, xs)                    
                    p = np.real(ys)**2 / (np.imag(ys)**2 + np.real(ys)**2) * (np.pi/2)
                    # print(np.real(ys))
                    if np.real(ys) > 0:    
                        cor[i][j][l] = p
                    else:
                        cor[i][j][l] = np.pi + p
                
                elif method == "Spectral":
                    xs = np.abs(np.fft.fft(x))
                    hs = np.abs(np.fft.fft(h))
                    ys = metrics.mutual_info_score(hs - np.mean(hs), xs - np.mean(xs))                    
                    sigms = 1 # np.std(hs)*np.std(xs)*hs.shape[0]
                    cor[i][j][l] = ys/sigms
                
                else: # Temporal
                    y = metrics.mutual_info_score(h - np.mean(h), x - np.mean(x))
                    sigm = 1 # np.std(h)*np.std(x)*h.shape[0]
                    cor[i][j][l] = y/sigm
        
    return cor            


def granger_causality(Temps, signals, method="Temporal", channels=None, Overlap=None, 
              filter_order=4, lbf=20, ubf=32, sampling_rate=500, lag=3):
    
    # chs = [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 18, 19]        
    chs = [] 
    if channels == None:
        for i in range(21):
            chs.append(i)
    else:
        chs = channels
    
    Interval_size = int(signals.shape[1]/Temps)
    cor = np.zeros([len(chs), len(chs), Temps])
        
    nyq = sampling_rate/2
    b, a = signal.butter(filter_order, [lbf/nyq, ubf/nyq], btype='band')
    for k in range(21):
        signals[k, :] = signal.lfilter(b, a, signals[k, :])
    
    for i in range(len(chs)): #signals.shape[0]):
        for j in range(len(chs)):
            for l in range(Temps):
                ie = chs[i]
                je = chs[j]
                h = signals[ie, l*Interval_size:(l+1)*Interval_size]
                x = signals[je, l*Interval_size:(l+1)*Interval_size]
                
                d = np.array((x, h)).transpose()
                gc = gct(d, lag, verbose=False)
                y = 0
                
                for lg in range(1, lag+1):
                    if gc[lg][0]['lrtest'][1] > 0:
                        y += gc[lg][0]['lrtest'][1]

                cor[i][j][l] = y
        
    return cor            


def heatmap(img, chs, signal_headers, mode="Normal", name="h", tit="", save=False):
    
    fig, ax = plt.subplots(figsize=(31, 21))
    im = ax.imshow(img)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(tit)
    cax = fig.add_axes([0.26, 0.91, 0.5, 0.05])
    fig.colorbar(im, cax=cax, orientation='horizontal')
    for i in range(len(chs)):
        ax.text(-1.5, i, signal_headers[chs[i]]['label'])
        ax.text(i-0.5, -0.7, signal_headers[chs[i]]['label'])
    if save == True:
        fig.savefig(name + ".png")
        plt.close()
    else:
        fig.show()   
    return
    

def graphmap(cor, chs, signal_headers, sensor_locs, mode=None, name="0",
             fname="Graphs/", titl=None, transp=False, directed=False):
    
    img = load_image("EEGMA/10-20.jpg")
    fig, ax = plt.subplots(figsize=(21, 23))
    if transp:
        cor = np.transpose(cor)
        
    w = img.shape[0]
    h = img.shape[1]

    lx = sensor_locs[:, 0]
    lx *= h/5
    ly = sensor_locs[:, 1]
    ly *= w/5
    
    for i in range(len(chs)):
        ax.text(sensor_locs[chs[i], 0], sensor_locs[chs[i], 1]
                , signal_headers[chs[i]]['label'], color='green', fontsize=30)
        for j in range(i+1, len(chs)):
            x = np.linspace(lx[chs[i]], lx[chs[j]], 10)
            y = np.linspace(ly[chs[i]], ly[chs[j]], 10)
            
            ax.plot(x, y, marker='.', linestyle='-', linewidth=cor[chs[i], chs[j]]*10 + 1, 
                    color=[1-cor[chs[i], chs[j]], 1-cor[chs[i], chs[j]],
                           1-cor[chs[i], chs[j]]])
            
            if directed:
                if transp:
                    ax.arrow(x[4], y[4], x[5]-x[4], y[5]-y[4], shape='full',
                             lw=0.1, length_includes_head=True,
                             head_width=0.1 + 3*cor[chs[i], chs[j]])
                else:
                    ax.arrow(x[5], y[5], x[4]-x[5], y[4]-y[5], shape='full',
                             lw=0.1, length_includes_head=True,
                             head_width=0.1 + 3*cor[chs[i], chs[j]])
    
    ax.set_title(titl)
    if mode == "save":
        fig.savefig("Graphs/" + fname + name + ".png")
        # fig.close()
        plt.close("all")
        print("closed")
    
    else:
        fig.show()   
    
    return 


def graphmat(cor, mode=None):
    
    cor = cor - np.min(np.min(cor))
    cor = cor / np.max(np.max(cor))
    # if mode == "Granger1" or mode == "Granger2":
    #     cor = 1 - cor
        
    return cor


def animate(folderpath=""):
    
    fp_in = folderpath
    fp_out = ""
    print(glob.glob(fp_in))
    img, *imgs = [load_image(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
    return
