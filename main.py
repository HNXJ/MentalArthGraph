from GraphMethods import *
from DeepFeature import *
from Clustering import *


from sklearn import metrics
import numpy as np
import pickle


def save_list(l, filename="List0.txt"):
        
    with open(filename, "wb") as f_temp:
        pickle.dump(l, f_temp)
    return


def load_list(filename="List0.txt"):
    
    with open(filename, "rb") as f_temp:
        l = pickle.load(f_temp)
    
    return l




# Channel initialization
chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
chs1 = []
for i in range(20): 
    chs1.append(i)
chs0 = chs1
frame = 3



# # Data loading and preprocessing: coherence
# ds, ds_pearson, ds_spectral = datasets_preparation(frames=frame, order=4, cf1=15,
#                                                     cf2=25, mth="coherence")
# ds_p0, ds_p1 = split_count_quality(ds_pearson, id1="0", id2="1")
# ds_s0, ds_s1 = split_count_quality(ds_spectral, id1="0", id2="1")




# # Data loading and preprocessing mutual information
# chs0 = chs1
# ds, ds_temp, ds_spect = datasets_preparation(frames=frame, order=4, cf1=15,
#                                                     cf2=25, mth="mutual_info")
# ds_t0, ds_t1 = split_count_quality(ds_temp, id1="0", id2="1")
# ds_f0, ds_f1 = split_count_quality(ds_spect, id1="0", id2="1")




# # Save lists for next time :!
# save_list(ds_p0, "Data/p0_" + str(frame) + "f_[15-25].txt")
# save_list(ds_p1, "Data/p1_" + str(frame) + "f_[15-25].txt")
# save_list(ds_s0, "Data/s0_" + str(frame) + "f_[15-25].txt")
# save_list(ds_s1, "Data/s1_" + str(frame) + "f_[15-25].txt")

# # Save lists for next time :D
# save_list(ds_t0, "Data/t0_" + str(frame) + "f_[15-25].txt")
# save_list(ds_t1, "Data/t1_" + str(frame) + "f_[15-25].txt")
# save_list(ds_f0, "Data/f0_" + str(frame) + "f_[15-25].txt")
# save_list(ds_f1, "Data/f1_" + str(frame) + "f_[15-25].txt")




## Load later:
# ds_f0 = load_list("Data/f0_10f_[15-25].txt")
# ds_f1 = load_list("Data/f1_10f_[15-25].txt")
# ds_s0 = load_list("Data/s0_10f_[15-25].txt")
# ds_s1 = load_list("Data/s1_10f_[15-25].txt")


# # TTest heatmaps (p-values and stats)
# for f in range(9):
#     s, p = visualize_ttest_heatmap(ds_f0, ds_f1, fs=f, ff=(f+1), mode="MI_Spect9Frame_15_25", save=True) 
#     try:
#         st += s
#         pt += p
#     except:
#         st = s
#         pt = p

# for f in range(9):
#     s, p = visualize_ttest_heatmap(ds_p0, ds_p1, fs=f, ff=(f+1), mode="CH_Spect9Frame_32_38", save=True) 
#     try:
#         st += s
#         pt += p
#     except:
#         st = s
#         pt = p

## Overall (cumulative results)
# Coherence.heatmap(pt, chs0, ds_f0[0].signal_headers, mode="Normal", name="h", tit="", save=False)
# Coherence.heatmap(pt, chs0, ds[0].signal_headers, mode="Normal", name="h", tit="", save=False)




## TSNE clustering



## Deep classification parts (colab recommended for this part)
# TTest data extract
# x, y = get_dataset_cor1(ds_s0, ds_s1, f=frame)
# x, y = get_dataset_cor2(ds_s0, ds_s1, f=frame)
x, y = get_dataset_cor3(ds_s0, ds_s1)
k = tsne_cluster(X=x, Y=y, components=2, visualize=True, iterations=2000)
k = tsne_cluster(X=x, Y=y, components=3, visualize=True, iterations=2000)
# in_shape = [1600]

# dataset2 = SampleDataset()
# dataset2.X = x[3:16, :]
# dataset2.Y = y[3:16]
# classifier1 = DeepModel2(input_size=in_shape, dataset=dataset2)

# classifier1.train(200, 10, val=0.2)
# print(classifier1.encoder(x), "\n", y)

