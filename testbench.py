from GraphMethods import *
from DeepFeature import *
import numpy as np


# Channel initialization
chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
chs1 = []
for i in range(20): 
    chs1.append(i)


# Data loading and preprocessing
# chs0 = chs1
# ds, ds_pearson, ds_spectral = datasets_preparation(frames=3, order=10, cf1=18, cf2=36)
# ds_p0, ds_p1 = split_count_quality(ds_pearson, id1="0", id2="1")
# ds_s0, ds_s1 = split_count_quality(ds_spectral, id1="0", id2="1")


## Visualize I
# run_graph_visualize(ds_spectral, mode="Spectral", split="count-quality")
# run_graph_visualize(ds_pearson, mode="Pearson", split="count-quality")
# run_heatmap_visualize(ds_p0, mode="Spectral", split="quality", f=3)
# run_heatmap_visualize(ds_p1, mode="Pearson", split="quality", f=3)


## Visualize II
# visualize_graph_modified(ds_pearson[4], chs=None, mode="Pearson", ql="")
# visualize_signal(ds[10], t1=0, t2=60, order=2, cf1=10, cf2=20, ch=0)
# visualize_ttest_heatmap(ds_s0, ds_s1, fs=25, ff=35, mode="Spectral", save=False)
# visualize_ttest_heatmap(ds_s0, ds_s1, fs=7, ff=25, mode="Spectral", save=False)

# TTest mean heatmaps
for f in range(3):
    s, p = visualize_ttest_heatmap(ds_s0, ds_s1, fs=f, ff=(f+1), mode="Spect3Frame", save=True) 
    try:
        st += s
        pt += p
    except:
        st = s
        pt = p

Coherence.heatmap(st, chs0, ds[0].signal_headers, mode="Normal", name="h", tit="", save=False)
Coherence.heatmap(pt, chs0, ds[0].signal_headers, mode="Normal", name="h", tit="", save=False)

## Save ttest heatmaps
# for f in range(3):
#     visualize_ttest_heatmap(ds_p0, ds_p1, fs=f, ff=f+1, mode="Pearson", save=True)  
# for f in range(3):
#     visualize_ttest_heatmap(ds_s0, ds_s1, fs=f, ff=f+1, mode="Spectral", save=True)  
    
## Deep classification parts

## TTest data extract
# x, y = get_dataset_cor(ds_s0, ds_s1)
# in_shape = [20, 20, 4]

# dataset2 = SampleDataset()
# dataset2.X = x
# dataset2.Y = y
# classifier1 = DeepModel2(input_size=in_shape, dataset=dataset2)

# classifier1.train(100, 5, val=0.3)

# EEG_SHAPE = (21, 31000, 1)
# dataset = PhysionetDataset(foldername="EEGMA/")
# deepmodel = DeepModel(EEG_SHAPE, dataset)
# deepmodel.train(100, 7)

# f = deepmodel.get_filters()
