from GraphMethods import *
from DeepFeature import *
import numpy as np


# Channel initialization
chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
chs1 = []
for i in range(20): 
    chs1.append(i)


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

# # Load later:
# ds_f0 = load_list("Data/f0_3f_[15-25].txt")
# ds_f1 = load_list("Data/f1_3f_[15-25].txt")
# ds_s0 = load_list("Data/s0_3f_[15-25].txt")
# ds_s1 = load_list("Data/s1_3f_[15-25].txt")




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




## Animator
# make_gif(path="Graphs/Spectral_Bad_count-quality_12_f_bandpass[60_120]Hz/",
#          fname="Animate/sb12_60_120.gif", duration=0.17)


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

# # TTest mean heatmaps
# for f in range(3):
#     s, p = visualize_ttest_heatmap(ds_s0, ds_s1, fs=f, ff=(f+1), mode="Spect3Frame", save=True) 
#     try:
#         st += s
#         pt += p
#     except:
#         st = s
#         pt = p

# Coherence.heatmap(st, chs0, ds[0].signal_headers, mode="Normal", name="h", tit="", save=False)
# Coherence.heatmap(pt, chs0, ds[0].signal_headers, mode="Normal", name="h", tit="", save=False)

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
