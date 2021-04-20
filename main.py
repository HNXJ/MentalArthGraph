from GraphMethods import *
from DeepFeature import *

from sklearn import metrics
import numpy as np


# Channel initialization
chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
chs1 = []
for i in range(20): 
    chs1.append(i)


# Data loading and preprocessing: coherence
chs0 = chs1
ds, ds_pearson, ds_spectral = datasets_preparation(frames=10, order=4, cf1=32,
                                                    cf2=38, mth="coherence")
ds_p0, ds_p1 = split_count_quality(ds_pearson, id1="0", id2="1")
ds_s0, ds_s1 = split_count_quality(ds_spectral, id1="0", id2="1")

# Data loading and preprocessing mutual information
chs0 = chs1
ds, ds_temp, ds_spect = datasets_preparation(frames=10, order=4, cf1=32,
                                                    cf2=38, mth="mutual_info")
ds_t0, ds_t1 = split_count_quality(ds_pearson, id1="0", id2="1")
ds_f0, ds_f1 = split_count_quality(ds_spectral, id1="0", id2="1")

for f in range(9):
    s, p = visualize_ttest_heatmap(ds_f0, ds_f1, fs=f, ff=(f+1), mode="MI_Spect4Frame_32_38", save=True) 
    try:
        st += s
        pt += p
    except:
        st = s
        pt = p

for f in range(9):
    s, p = visualize_ttest_heatmap(ds_p0, ds_p1, fs=f, ff=(f+1), mode="CH_Spect4Frame_32_38", save=True) 
    try:
        st += s
        pt += p
    except:
        st = s
        pt = p

# Coherence.heatmap(st, chs0, ds[0].signal_headers, mode="Normal", name="h", tit="", save=False)
# Coherence.heatmap(pt, chs0, ds[0].signal_headers, mode="Normal", name="h", tit="", save=False)

## Deep classification parts

# TTest data extract
# x, y = get_dataset_cor2(ds_s0, ds_s1, 4)
# in_shape = [1600]

# dataset2 = SampleDataset()
# dataset2.X = x[3:16, :]
# dataset2.Y = y[3:16]
# classifier1 = DeepModel2(input_size=in_shape, dataset=dataset2)

# classifier1.train(200, 10, val=0.2)
# print(classifier1.encoder(x), "\n", y)