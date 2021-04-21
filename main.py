from GraphMethods import *
from DeepFeature import *
from Clustering import *


from sklearn import metrics
import numpy as np
import pickle


def save_list(l, filename="List0.txt"):
        
    with open(filename, "wb") as f_temp:
        pickle.dump(l, fp)
    return


def load_list(filename="List0.txt"):
    
    with open(filename, "rb") as f_temp:
        l = pickle.load(fp)
    
    return l




# Channel initialization
chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
chs1 = []
for i in range(20): 
    chs1.append(i)
chs0 = chs1




# Data loading and preprocessing: coherence
ds, ds_pearson, ds_spectral = datasets_preparation(frames=10, order=4, cf1=15,
                                                    cf2=25, mth="coherence")
ds_p0, ds_p1 = split_count_quality(ds_pearson, id1="0", id2="1")
ds_s0, ds_s1 = split_count_quality(ds_spectral, id1="0", id2="1")




# Data loading and preprocessing mutual information
chs0 = chs1
ds, ds_temp, ds_spect = datasets_preparation(frames=10, order=4, cf1=15,
                                                    cf2=25, mth="mutual_info")
ds_t0, ds_t1 = split_count_quality(ds_temp, id1="0", id2="1")
ds_f0, ds_f1 = split_count_quality(ds_spect, id1="0", id2="1")




# Save lists for next time!
save_list(ds_p0, "p0_10f_[15-25]")
save_list(ds_p1, "p1_10f_[15-25]")
save_list(ds_s0, "s0_10f_[15-25]")
save_list(ds_s1, "s1_10f_[15-25]")

# Save lists for next time :D
save_list(ds_t0, "t0_10f_[15-25]")
save_list(ds_t1, "p1_10f_[15-25]")
save_list(ds_f0, "f0_10f_[15-25]")
save_list(ds_f1, "p1_10f_[15-25]")




# Load later:
ds_f0 = load_list("f0_10f_[15-25]")
ds_f1 = load_list("f1_10f_[15-25]")


## TTest heatmaps (p-values and stats)
# for f in range(9):
#     s, p = visualize_ttest_heatmap(ds_f0, ds_f1, fs=f, ff=(f+1), mode="MI_Spect9Frame_32_38", save=True) 
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
# Coherence.heatmap(st, chs0, ds[0].signal_headers, mode="Normal", name="h", tit="", save=False)
# Coherence.heatmap(pt, chs0, ds[0].signal_headers, mode="Normal", name="h", tit="", save=False)




## TSNE clustering



## Deep classification parts (colab recommended for this part)
# TTest data extract
# x, y = get_dataset_cor2(ds_s0, ds_s1, 4)
# in_shape = [1600]

# dataset2 = SampleDataset()
# dataset2.X = x[3:16, :]
# dataset2.Y = y[3:16]
# classifier1 = DeepModel2(input_size=in_shape, dataset=dataset2)

# classifier1.train(200, 10, val=0.2)
# print(classifier1.encoder(x), "\n", y)

