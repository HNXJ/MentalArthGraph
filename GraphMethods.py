from matplotlib import pyplot as plt
from scipy import signal, stats
import statsmodels.api as sm

import numpy as np
import statsmodels
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
    
    def set_cor_mutual_info(self, frames, mode, chs, order, cf1, cf2, sr):
        
        self.cor = cor = Coherence.mutual_info(frames, self.signals, mode, chs, 
                                             None, order, cf1, cf2, sr)
        self.cf1 = cf1
        self.cf2 = cf2
        return
    
    def set_cor_granger_causality(self, frames, mode, chs, order, cf1, cf2, sr, lag):
        
        self.cor = cor = Coherence.granger_causality(frames, self.signals, mode, chs, 
                                             None, order, cf1, cf2, sr, lag)
        self.cf1 = cf1
        self.cf2 = cf2
        return
    

def load_dataset(cnt=35, fname="EEGMA/"):
 
    data = []
    chs1 = []
    for i in range(20):
        chs1.append(i)
    
    for q in range(cnt):

        signals, signal_headers, header, fields, rows = Coherence.load_file(q, foldername=fname)                
        data.append(subject_eeg_arithmetics(signals, None, rows[q][4],
                                            rows[q][5], signal_headers, 
                                            rows[q][0]))
    
    return data


def visualize_signal(ds, t1=0, t2=60, order=2, cf1=10, cf2=20, ch=0):
    
    n = (t2-t1)*500
    t = np.linspace(t1*500, t2*500, n)
    nyq = 500/2
    b, a = signal.butter(order, [cf1/nyq, cf2/nyq], btype='band')
    Q = signal.lfilter(b, a, ds.signals[ch, t1*500:t2*500])
    
    plt.figure(figsize=(33, 15))
    plt.subplot(3, 1, 1)
    plt.plot(t, Q)
    plt.title(ds.signal_headers[ch]['label'])
    plt.xlabel('Time (mS)')
    plt.ylabel('Amp')
    plt.grid(True)
    
    Ft = np.abs(np.fft.fft(Q))
    f = np.linspace(0, 250, np.int(n/2))
    plt.subplot(3, 1, 2)
    plt.plot(f, Ft[0:np.int(n/2)])
    plt.title('Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.grid(True)
    
    f = np.linspace(0, 500/13, np.int(n/13))
    plt.subplot(3, 1, 3)
    plt.plot(f, Ft[0:np.int(n/13)])
    plt.title('Magnified spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.grid(True)
    plt.show()
    return


def calculate_graph(idb=0, idf=1, frames=30, chs=None, mode="Pearson"):
    
    chs1 = []
    for i in range(20):
        chs1.append(i)
    
    cors = np.zeros([idf-idb+1, 20, 20, frames])
    for q in range(idb, idf+1):
        
        fname = "g_" + mode + "_" + str(q) + "_" + str(frames) + "f"
        try:
            os.mkdir("Graphs/" + fname)
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
            os.mkdir("Graphs/" + fname)
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


def visualize_graph_modified(subject_set, chs=None, mode="Pearson", ql=None,
                             transp=False, directed=False):
    
    chs1 = []
    for i in range(20):
        chs1.append(i)
        
    if chs == None:
        chs = chs1
  
    fname = mode + "_" + str(subject_set.id) + "_" + str(subject_set.cor.shape[2]) + "_f" + "_bandpass[" + str(subject_set.cf1) + "_" + str(subject_set.cf2) + "]Hz"
    try:
        os.mkdir("Graphs/" + fname)
    except:
        pass
    
    title_subj = "_" + str(subject_set.quality)
    print("### Calculating graphs, ...")
    
    for frame in range(subject_set.cor.shape[2]):
        
        print("frame no." + str(frame) + " ... ")
        sensor_locs1 = Coherence.get_sensor_locs(mode="EEG")
        cor_a = Coherence.graphmat(subject_set.cor[:, :, frame], mode=mode)
        Coherence.graphmap(cor_a, chs, subject_set.signal_headers,
                            sensor_locs=sensor_locs1,
                            mode="save", name="f_" + str(frame) + "count_" +
                            str(subject_set.subtractions),
                            fname=fname + "/", 
                            titl=title_subj + "_frame_" + str(frame) + 
                            "_Correlation_" + mode + "_" + ql, transp=transp,
                            directed=directed)
    
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
    
    ds_0 = []
    ds_1 = []
    
    for q in range(len(ds)):

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
    

def visualize_mean_graph(ds, chs0=None, split_type="quality", mode=None, transp=False,
                         directed=False):
    
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
        
    visualize_graph_modified(ds0, chs=chs0, mode=mode, ql=lb1, transp=transp,
                             directed=directed)
    visualize_graph_modified(ds1, chs=chs0, mode=mode, ql=lb2, transp=transp,
                             directed=directed)


def run_graph_visualize(ds, mode=None, split=None, transp=False, directed=False):
    
    chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
    chs1 = []
    for i in range(20): 
        chs1.append(i)
    
    chs0 = chs1
    print("Stablishing sets, ")
    visualize_mean_graph(ds, chs0, split_type=split, mode=mode, transp=transp,
                         directed=directed)
    return

    
def run_heatmap_visualize(ds, mode=None, split=None, f=0):
    
    chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
    chs1 = []
    for i in range(20): 
        chs1.append(i)
    
    chs0 = chs1
    Coherence.heatmap(ds[0].cor[:, :, f], chs0, ds[0].signal_headers)
    return


def datasets_preparation(frames=10, cnt=36, order=4, cf1=15, cf2=25, fname="EEGMA/", mth="coherence", lag=3):
    
    chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
    chs1 = []
    for i in range(20): 
        chs1.append(i)
    
    chs0 = chs1
    
    ds = load_dataset(cnt=cnt, fname=fname)
    print("Stablishing sets, " + mth)
    ds_pearson = ds
    ds_spectral = ds
    
    if mth == "coherence":
        for i in range(cnt):
            ds_pearson[i].set_cor(frames=frames, mode="Pearson", chs=chs0, order=order,
                                  cf1=cf1, cf2=cf2, sr=500)
            if i%5 == 0:
                print("Preprocessing subject no." + str(i))
        
        print("Pearson done.")
        for i in range(cnt):
            ds_spectral[i].set_cor(frames=frames, mode="Spectral", chs=chs0, order=order,
                                   cf1=cf1, cf2=cf2, sr=500)
            if i%5 == 0:
                print("Preprocessing subject no." + str(i))
        
        print("Spectral done.")
    
    elif mth == "mutual_info":
        for i in range(cnt):
            ds_pearson[i].set_cor_mutual_info(frames=frames, mode="Pearson", chs=chs0, order=order,
                                  cf1=cf1, cf2=cf2, sr=500)
            if i%5 == 0:
                print("Preprocessing subject no." + str(i))
        
        print("Pearson done.")
        for i in range(cnt):
            ds_spectral[i].set_cor_mutual_info(frames=frames, mode="Spectral", chs=chs0, order=order,
                                   cf1=cf1, cf2=cf2, sr=500)
            if i%5 == 0:
                print("Preprocessing subject no." + str(i))
        
        print("Spectral done.")
        
    elif mth == "granger":
        print("Granger method selected, this takes too long to be done, so progress will be printed.")
        for i in range(cnt):
            ds_pearson[i].set_cor_granger_causality(frames=frames, mode="Pearson", chs=chs0, order=order,
                                  cf1=cf1, cf2=cf2, sr=500, lag=lag)
            
            print(i, "---=Done=---========================================---")
            if i%5 == 0:
                print("Preprocessing subject no." + str(i))
        
        print("Granger done.")

    return ds, ds_pearson, ds_spectral


def split_quality(ds, id1="0", id2="1"):
    
    ds_1 = []
    ds_2 = []
    
    for q in range(35):

        if ds[q].quality == "0":
            
            ds_1.append(ds[q])
            
        else:
        
            ds_2.append(ds[q])
            
    return ds_1, ds_2

    
def split_count_quality(ds, id1="0", id2="1"):
    
    ds_1 = []
    ds_2 = []
    
    for q in range(35):

        if ds[q].quality == "0" and float(ds[q].subtractions) < 10.0:
            
            ds_1.append(ds[q])
            
        elif float(ds[q].subtractions) > 22.5:
        
            ds_2.append(ds[q])
                
    return ds_1, ds_2


def ttest_prep(x, y):
    
    n = x.shape[0]*x.shape[1]
    m = y.shape[0]*y.shape[1]
    x = np.reshape(x, [n, -1])
    y = np.reshape(y, [m, -1])
    x1 = x
    y1 = y
    
    for i in range(m-1):
        x1 = np.concatenate([x1, x], 0)
    for i in range(n-1):
        y1 = np.concatenate([y1, y], 0)
        
    return x1, y1
    
def ttest_heatmap(ds1=None, ds2=None, fs=0, ff=0):
    
    x = np.zeros([20, 20, ds1[0].cor.shape[2], len(ds1)])
    y = np.zeros([20, 20, ds2[0].cor.shape[2], len(ds2)])
    img_s = np.zeros([20, 20])
    img_p = np.zeros([20, 20])
    
    for i in range(len(ds1)):
        x[:, :, :, i] = ds1[i].cor
    for i in range(len(ds2)):
        y[:, :, :, i] = ds2[i].cor
    
    for i in range(20):
        for j in range(20):
            if not i == j:
                x1, y1 = ttest_prep(x[i, j, fs:ff, :], y[i, j, fs:ff, :])
                # img_s[i, j], img_p[i, j] = stats.ttest_ind(x1, y1, equal_var=False)
                # img_s[i, j], img_p[i, j] = sm.stats.ztest(x1, y1)
                img_s[i, j], img_p[i, j], _ = sm.stats.ttest_ind(x1, y1)
            elif i == j:
                img_s[i, i] = 0
                img_p[i, i] = 0
            
    return img_s, img_p


def ttest_heatmap_multiframe(ds1=None, ds2=None, fs=0, ff=2): # Incomplete
    
    x = np.zeros([20, 20, ds1[0].cor.shape[2], len(ds1)])
    y = np.zeros([20, 20, ds2[0].cor.shape[2], len(ds2)])
    img_s = np.zeros([20, 20])
    img_p = np.zeros([20, 20])
    
    for i in range(len(ds1)):
        x[:, :, :, i] = ds1[i].cor
    for i in range(len(ds2)):
        y[:, :, :, i] = ds2[i].cor
    
    for i in range(20):
        for j in range(20):
            if not i == j:
                x1, y1 = ttest_prep(x[i, j, fs:ff, :], y[i, j, fs:ff, :])
                # img_s[i, j], img_p[i, j] = sm.stats.ttest_rel(x1, y1, equal_var=False)
            
    return img_s, img_p
    

def visualize_ttest_heatmap(ds1, ds2, fs=0, ff=0 , mode="", save=False, render=True):
    
    chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
    chs1 = []
    for i in range(20): 
        chs1.append(i)
    
    chs0 = chs1
    s, p = ttest_heatmap(ds1, ds2, fs, ff)
    
    try: 
        os.mkdir("Tests/")
        try:
            os.mkdir("Tests/TTest_" + mode)
        except:
            pass
    except:
        try:
            os.mkdir("Tests/TTest_" + mode)
        except:
            pass
    
    if render:
        Coherence.heatmap(s, chs0, ds1[0].signal_headers,
                          name="Tests/TTest_" + mode + "/stat_fs_" + str(fs) + "_ff_" + str(ff),
                          tit="stat", save=save)
        Coherence.heatmap(p, chs0, ds2[0].signal_headers,
                          name="Tests/TTest_" + mode + "/pval_fs_" + str(fs) + "_ff_" + str(ff),
                          tit="pval", save=save)
    return s, p


def get_dataset_cor_multiframe(ds1=None, ds2=None, f=1):
    
    # x = np.zeros([20, 20, ds1[0].cor.shape[2], len(ds1)])
    # y = np.zeros([20, 20, ds2[0].cor.shape[2], len(ds2)])
    x = np.zeros([(len(ds1) + len(ds2)), 20 * 20 * f])
    y = np.zeros([(len(ds1) + len(ds2)), 2])
    
    cnt = 0
    for i in range(len(ds1)):
        x[cnt, :] = np.reshape(ds1[i].cor[:, :, :], [1, -1])
        y[cnt, 0] = 1
        # y[cnt, 1] = 0
        cnt += 1
        
    for i in range(len(ds2)):
        x[cnt, :] = np.reshape(ds2[i].cor[:, :, :], [1, -1])
        y[cnt, 1] = 1
        # y[cnt, 1] = 0
        cnt += 1

    
    return x, y


def get_dataset_cor_frame_augmented(ds1=None, ds2=None, f=1):
    
    # x = np.zeros([20, 20, ds1[0].cor.shape[2], len(ds1)])
    # y = np.zeros([20, 20, ds2[0].cor.shape[2], len(ds2)])
    x = np.zeros([(len(ds1) + len(ds2))*f, 20 * 20])
    y = np.zeros([(len(ds1) + len(ds2))*f, 2])
    
    cnt = 0
    for i in range(len(ds2)):
        for j in range(f):
            x[cnt*f + j, :] = np.reshape(ds2[i].cor[:, :, j], [1, -1])
            y[cnt*f + j, 0] = 1
            # y[cnt, 1] = 0
        cnt += 1
        
    for i in range(len(ds1)):
        for j in range(f):
            x[cnt*f + j, :] = np.reshape(ds1[i].cor[:, :, j], [1, -1])
            y[cnt*f + j, 1] = 1
            # y[cnt, 1] = 0
        cnt += 1

    
    return x, y


def get_dataset_cor_meanframe(ds1=None, ds2=None):
    
    x = np.zeros([(len(ds1) + len(ds2)), 20 * 20])
    y = np.zeros([(len(ds1) + len(ds2)), 2])
    
    cnt = 0
    for i in range(len(ds2)):
        cor_temp = np.mean(ds2[i].cor[:, :, :], 2)
        x[cnt, :] = np.reshape(cor_temp, [1, -1])
        y[cnt, 0] = 1
        cnt += 1
        
    for i in range(len(ds1)):
        cor_temp = np.mean(ds1[i].cor[:, :, :], 2)
        x[cnt, :] = np.reshape(cor_temp, [1, -1])
        y[cnt, 1] = 1
        cnt += 1

    return x, y


def get_dataset_cor_selective(ds1=None, ds2=None, edges=None):
    
    x = np.zeros([(len(ds1) + len(ds2)), len(edges)])
    y = np.zeros([(len(ds1) + len(ds2)), 2])
    
    cnt = 0
    for i in range(len(ds2)):
        cor_temp = np.mean(ds2[i].cor[:, :, :], 2)
        cor_temp = np.reshape(cor_temp, [1, -1])
        cor_temp = cor_temp[0, edges]
        
        x[cnt, :] = np.reshape(cor_temp, [1, -1])
        y[cnt, 0] = 1
        cnt += 1
        
    for i in range(len(ds1)):
        cor_temp = np.mean(ds1[i].cor[:, :, :], 2)
        cor_temp = np.reshape(cor_temp, [1, -1])
        cor_temp = cor_temp[0, edges]
        
        x[cnt, :] = np.reshape(cor_temp, [1, -1])
        y[cnt, 1] = 1
        cnt += 1

    return x, y


def select_electrodes(table=None, thresh=0.1):
    
    edges = []
    for i in range(table.shape[0]):
        for j in range(i+1, table.shape[1]):
            if table[i, j] < thresh:
                edges.append(i + j*table.shape[1])
                
    return edges


def correction_fdr_test(pvalues=None, alpha=0.1, method='indep',
                        is_sorted=False, verbose=False, headers=None):
    
    a = np.reshape(pvalues, [-1])
    d, pv_c = statsmodels.stats.multitest.fdrcorrection(pvals=a, alpha=alpha,
                                                        method=method, is_sorted=is_sorted)
    
    m = pvalues.shape[0]
    n = d.shape[0]
    
    for i in range(m):
        d[i*m+i] = False
        pv_c[i*m+i] = 1.0
    
    if verbose:
        print(" %d True of %d, Alpha = %f. Significant connections and their modified p-value: " % (np.sum(d), n, alpha))
        # print(headers)
        for i in range(n):
            if d[i]:
                print(headers[np.int(i/m)]['label'], " - ",
                      headers[np.int(i%m)]['label'], " p-value = ", pv_c[i])
        
    d = np.reshape(d, pvalues.shape)
    pv_c = np.reshape(pv_c, pvalues.shape)
    
    return d, pv_c

