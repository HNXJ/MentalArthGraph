# import scipy
import numpy as np
# from scipy import signal
# from matplotlib import pyplot as plt
import Coherence
import os


class subject_eeg_arithmetics:
    
    def __init__(self, signals, cor, subtractions, quality, signal_headers, id, cf1=None, cf2=None):
        
        self.signals = signals
        self.cor = cor
        self.subtractions = subtractions
        self.quality = quality
        
        self.signal_headers = signal_headers
        self.id = id
        self.cf1 = cf1
        self.cf2 = cf2
        return 
    
    def set_cor(self, frames, mode, chs, order, cf1, cf2, sr):
        
        self.cor = cor = Coherence.coherence(frames, self.signals, mode, chs, 
                                             None, order, cf1, cf2, sr)
        self.cf1 = cf1
        self.cf2 = cf2
        return


def load_dataset(cnt=35):
 
    data = []
    chs1 = []
    for i in range(20):
        chs1.append(i)
    
    for q in range(cnt):
        signals, signal_headers, header, fields, rows = Coherence.load_file(q)                
        data.append(subject_eeg_arithmetics(signals, None, rows[q][4],
                                            rows[q][5], signal_headers, 
                                            rows[q][0]))
    
    return data


def calculate_graph(idb=0, idf=1, frames=30, chs=None, mode="Pearson"):
    
    chs1 = []
    for i in range(20):
        chs1.append(i)
    
    cors = np.zeros([idf-idb+1, 20, 20, frames])
    for q in range(idb, idf+1):
        
        fname = "g_" + mode + "_" + str(q) + "_" + str(frames) + "f"
        try:
            os.mkdir("Graph/" + fname)
        except:
            pass
        
        signals, signal_headers, header, fields, rows = Coherence.load_file(q)                
        cor = Coherence.coherence(frames, signals, mode, chs1, None, 4, 24, 36, 500)
        title_subj = str(fields[4]) + "_" + str(rows[q][0]) + "_" + str(rows[q][4])
        print("### Calculating graphs, ...")
        
        for k in range(frames):
            
            print("frame no." + str(k) + " ... ")
            sensor_locs1 = Coherence.get_sensor_locs(mode="EEG")
            cor[:, :, k] = Coherence.graphmat(cor[:, :, k])
        
        print("ID no." + str(q) + " -> Finished.")
        cors[q-idb, :, :, :] = cor

    return cor, fields, rows


def visualize_graph(idb=0, idf=1, frames=30, chs=None, mode="Pearson"):
    
    chs1 = []
    for i in range(20):
        chs1.append(i)
    
    for q in range(idb, idf+1):
        
        fname = "g_" + mode + "_" + str(q) + "_" + str(frames) + "f"
        try:
            os.mkdir("Graph/" + fname)
        except:
            pass
        
        signals, signal_headers, header, fields, rows = Coherence.load_file(q)                
        cor = Coherence.coherence(frames, signals, mode, chs1, None, 4, 24, 36, 500)
        title_subj = str(fields[4]) + "_" + str(rows[q][0]) + "_" + str(rows[q][4])
        print("### Calculating graphs, ...")
        
        for k in range(frames):
            
            print("frame no." + str(k) + " ... ")
            sensor_locs1 = Coherence.get_sensor_locs(mode="EEG")
            cor_a = Coherence.graphmat(cor[:, :, k])
            Coherence.graphmap(cor_a, chs, signal_headers,
                               sensor_locs=sensor_locs1,
                                mode="save", name="f_" + str(k) + "count_" +
                                str(rows[q][4]),
                                fname=fname + "/", 
                                titl=title_subj + "_frame_" + str(k) + 
                                "_Correlation_" + mode)
        
        print("ID no." + str(q) + " -> Finished.")

    return


def visualize_graph_modified(subject_set, chs=None, mode="Pearson", ql=None):
    
    chs1 = []
    for i in range(20):
        chs1.append(i)
  
    fname = mode + "_" + str(subject_set.id) + "_" + str(subject_set.cor.shape[2]) + "_f" + "_bandpass[" + str(subject_set.cf1) + "_" + str(subject_set.cf2) + "]Hz"
    try:
        os.mkdir("Graph/" + fname)
    except:
        pass
    
    title_subj = "_" + str(subject_set.quality)
    print("### Calculating graphs, ...")
    
    for frame in range(subject_set.cor.shape[2]):
        
        print("frame no." + str(frame) + " ... ")
        sensor_locs1 = Coherence.get_sensor_locs(mode="EEG")
        cor_a = Coherence.graphmat(subject_set.cor[:, :, frame])
        Coherence.graphmap(cor_a, chs, subject_set.signal_headers,
                            sensor_locs=sensor_locs1,
                            mode="save", name="f_" + str(frame) + "count_" +
                            str(subject_set.subtractions),
                            fname=fname + "/", 
                            titl=title_subj + "_frame_" + str(frame) + 
                            "_Correlation_" + mode + "_" + ql)
    
    return


def mean_graph_quality(ds, id1="0", id2="1"):
    
    ds_1 = []
    ds_2 = []
    
    for q in range(35):

        if ds[q].quality == "0":
            
            try:
                ds_0.signals += ds[q].signals
                ds_0.cor += ds[q].cor
                ds_0.quality += 1
                ds_0.id = id1
                
            except:
                ds_0 = subject_eeg_arithmetics(ds[q].signals, ds[q].cor, ds[q].subtractions,
                                            1, ds[q].signal_headers, 
                                            id1, ds[q].cf1, ds[1].cf2)
            
        else:
        
            try:
                ds_1.signals += ds[q].signals
                ds_1.cor += ds[q].cor
                ds_1.quality += 1
                ds_1.id = id2
                
            except:
                ds_1 = subject_eeg_arithmetics(ds[q].signals, ds[q].cor, ds[q].subtractions,
                                            1, ds[q].signal_headers, 
                                            id2, ds[q].cf1, ds[1].cf2)
                
    ds_0.cor /= ds_0.quality
    ds_1.cor /= ds_1.quality
    return ds_0, ds_1

    
def mean_graph_count_quality(ds, id1="0", id2="1"):
    
    ds_1 = []
    ds_2 = []
    
    for q in range(35):

        if ds[q].quality == "0" and float(ds[q].subtractions) < 9.0:
            
            try:
                ds_0.signals += ds[q].signals
                ds_0.cor += ds[q].cor
                ds_0.quality += 1
                ds_0.id = id1
                
            except:
                ds_0 = subject_eeg_arithmetics(ds[q].signals, ds[q].cor, ds[q].subtractions,
                                            1, ds[q].signal_headers, 
                                            id1, ds[q].cf1, ds[1].cf2)
            
        elif float(ds[q].subtractions) > 18.0:
        
            try:
                ds_1.signals += ds[q].signals
                ds_1.cor += ds[q].cor
                ds_1.quality += 1
                ds_1.id = id2
                
            except:
                ds_1 = subject_eeg_arithmetics(ds[q].signals, ds[q].cor, ds[q].subtractions,
                                            1, ds[q].signal_headers, 
                                            id2, ds[q].cf1, ds[1].cf2)
                
    ds_0.cor /= ds_0.quality
    ds_1.cor /= ds_1.quality
    return ds_0, ds_1
    

def visualize_mean_graph(ds, chs0=None, split_type="quality", mode=None):
    
    chs1 = []
    for i in range(20): 
        chs1.append(i)
    
    print("Frames will be saved in folder <Graph> among this file")
    lb1 = "Bad_" + split_type
    lb2 = "Good_" + split_type
    
    if split_type == "quality":
        ds0, ds1 = mean_graph_quality(ds, lb1, lb2)
    if split_type == "count-quality":
        ds0, ds1 = mean_graph_count_quality(ds, lb1, lb2)
        
    visualize_graph_modified(ds0, chs=chs0, mode=mode, ql=lb1)
    visualize_graph_modified(ds1, chs=chs0, mode=mode, ql=lb2)


def run1(ds):
    
    chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
    chs1 = []
    for i in range(20): 
        chs1.append(i)
    
    chs0 = chs1
    print("Stablishing sets, ")
    ds_pearson = ds
    ds_spectral = ds
    
    for i in range(36):
        ds_pearson[i].set_cor(frames=4, mode="Pearson", chs=chs0, order=4, cf1=15, cf2=25, sr=500)
    
    for i in range(36):
        ds_spectral[i].set_cor(frames=4, mode="Spectral", chs=chs0, order=4, cf1=15, cf2=25, sr=500)
    
    visualize_mean_graph(ds_spectral, chs0, split_type="count-quality", mode="Spectral")
    visualize_mean_graph(ds_pearson, chs0, split_type="count-quality", mode="Pearson")
    # visualize_mean_graph(ds_spectral, chs2, mode="Spectral")
    # visualize_mean_graph(ds_pearson, chs2, mode="Pearson")
    
    # visualize_graph(16, 18, frames=5, chs=chs0, mode="Pearson")
    # Coherence.animate(folderpath="")
    
    
def run2(ds):
    
    chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
    chs1 = []
    for i in range(20): 
        chs1.append(i)
    
    chs0 = chs1
    # ds = load_dataset(cnt=36)
    # run1(ds)
    for i in range(1):
        ds[i].set_cor(frames=6, mode="Spectral", chs=chs0, order=4, cf1=15, cf2=25, sr=500)
    
    Coherence.heatmap(ds[0].cor[:, :, 4], chs0, ds[0].signal_headers)
    # print(ds[0].cor)
    # for i in range(36):
    #     print(ds_pearson[i].subtractions, ds_pearson[i].quality)    


def datasets_preparation(frames=10, cnt=36):
    
    chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
    chs1 = []
    for i in range(20): 
        chs1.append(i)
    
    chs0 = chs1
    
    ds = load_dataset(cnt=cnt)
    print("Stablishing sets, ")
    ds_pearson = ds
    ds_spectral = ds
    
    for i in range(cnt):
        ds_pearson[i].set_cor(frames=frames, mode="Pearson", chs=chs0, order=4, cf1=15, cf2=25, sr=500)
    
    print("Pearson done.")
    for i in range(cnt):
        ds_spectral[i].set_cor(frames=frames, mode="Spectral", chs=chs0, order=4, cf1=15, cf2=25, sr=500)
    
    print("Spectral done.")
    return ds, ds_pearson, ds_spectral


ds, ds_pearson, ds_spectral = datasets_preparation(frames=10)
ds_p0, ds_p1 = mean_graph_count_quality(ds_pearson, id1="0", id2="1")
ds_s0, ds_s1 = mean_graph_count_quality(ds_spectral, id1="0", id2="1")

# ds_q0 = np.zeros([20, 20, 10, ])    
