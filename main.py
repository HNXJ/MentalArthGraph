from GraphMethods import *
from DeepFeature import *
from Clustering import *
from Action import *


import numpy as np




# Channel initialization
chs2 = [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17]
chs1 = []
for i in range(20): 
    chs1.append(i)
chs0 = chs1


# Data loading and preprocessing: coherence
# ds, ds_pearson, ds_spectral = datasets_preparation(frames=frame, order=4, cf1=15,
                                                    # cf2=25, mth="coherence")
# ds_p0, ds_p1 = split_count_quality(ds_pearson, id1="0", id2="1")
# ds_s0, ds_s1 = split_count_quality(ds_spectral, id1="0", id2="1")

# # # Data loading and preprocessing: mutual information
# # chs0 = chs1
# # ds, ds_temp, ds_spect = datasets_preparation(frames=frame, order=4, cf1=15,
# #                                                     cf2=25, mth="mutual_info")
# # ds_t0, ds_t1 = split_count_quality(ds_temp, id1="0", id2="1")
# # ds_f0, ds_f1 = split_count_quality(ds_spect, id1="0", id2="1")


# # Saving before split to two labels
# save_list(ds_pearson, "Data/pearson_90f_[15-25].txt")
# save_list(ds_spectral, "Data/spectral_90f_[15-25].txt")


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
# ds_spectral = load_list("Data/spectral_90f_[15-25].txt")
# frame = 3

# ds_f0 = load_list("Data/f0_" + str(frame) + "f_[15-25].txt")
# ds_f1 = load_list("Data/f1_" + str(frame) + "f_[15-25].txt")
# ds_s0 = load_list("Data/s0_" + str(frame) + "f_[15-25].txt")
# ds_s1 = load_list("Data/s1_" + str(frame) + "f_[15-25].txt")




# # TTest heatmaps (p-values and stats)
# for f in range(frame):
#     s, p = visualize_ttest_heatmap(ds_s0, ds_s1, fs=f, ff=(f+1), mode="SC_Spect3Frame_15_25", save=True) 
#     try:
#         st += s
#         pt_sc += p
#     except:
#         st = s
#         pt_sc = p

# for f in range(frame):
#     s, p = visualize_ttest_heatmap(ds_f0, ds_f1, fs=f, ff=(f+1), mode="MI_Spect3Frame_15_25", save=True) 
#     try:
#         st += s
#         pt_mi += p
#     except:
#         st = s
#         pt_mi = p
    
    
# Overall (cumulative results)
th = 0.012
# Coherence.heatmap(pt_mi < th, chs0, ds_s0[0].signal_headers, mode="Normal", name="h", tit="MI", save=False)
# Coherence.heatmap(pt_sc < th, chs0, ds_s0[0].signal_headers, mode="Normal", name="h", tit="SC", save=False)
edges_mi = select_electrodes(pt_mi, th)
edges_sc = select_electrodes(pt_sc, th)
print(len(edges_mi), len(edges_sc))


## Clustering
# TTest data extract
# x, y = get_dataset_cor_multiframe(ds_s0, ds_s1, f=frame)
# x, y = get_dataset_cor_frame_augmented(ds_s0, ds_s1, f=frame)
# x, y = get_dataset_cor_meanframe(ds_s0, ds_s1)
# k = pca_cluster(X=x, Y=y, components=2, visualize=True, tit="PCA-2",
#                 save=True, name="sc_pca2")
# k = pca_cluster(X=x, Y=y, components=3, visualize=True, tit="PCA-3",
#                 save=True, name="sc_pca3")
# k = tsne_cluster(X=x, Y=y, components=2, visualize=True, iterations=5000,
#                   tit="TSNE-2", save=True, name="sc_tsne2")
# k = tsne_cluster(X=x, Y=y, components=3, visualize=True, iterations=5000,
#                   tit="TSNE-3", save=True, name="sc_tsne3")

# x, y = get_dataset_cor3(ds_f0, ds_f1)
# k = pca_cluster(X=x, Y=y, components=2, visualize=True, tit="PCA-2",
#                 save=True, name="mi_pca2")
# k = pca_cluster(X=x, Y=y, components=3, visualize=True, tit="PCA-3",
#                 save=True, name="mi_pca3")
# k = tsne_cluster(X=x, Y=y, components=2, visualize=True, iterations=5000,
#                   tit="TSNE-2", save=True, name="mi_tsne2")
# k = tsne_cluster(X=x, Y=y, components=3, visualize=True, iterations=5000,
#                   tit="TSNE-3", save=True, name="mi_tsne3")

 
# # Graph weights selective clustering
edges = edges_mi
x, y = get_dataset_cor_selective(ds_f0, ds_f1, edges)

k = pca_cluster(X=x, Y=y, components=2, visualize=True, tit="PCA-2",
                save=True, name="mi_pca2_selective")
k = pca_cluster(X=x, Y=y, components=3, visualize=True, tit="PCA-3",
                save=True, name="mi_pca3_selective")
k = tsne_cluster(X=x, Y=y, components=2, visualize=True, iterations=5000,
                  tit="TSNE-2", save=True, name="mi_tsne2_selective")
k = tsne_cluster(X=x, Y=y, components=3, visualize=True, iterations=5000,
                  tit="TSNE-3", save=True, name="mi_tsne3_selective")


# edges = edges_sc
# x, y = get_dataset_cor_selective(ds_s0, ds_s1, edges)

# k = pca_cluster(X=x, Y=y, components=2, visualize=True, tit="PCA-2",
#                 save=True, name="sc_pca2_selective")
# k = pca_cluster(X=x, Y=y, components=3, visualize=True, tit="PCA-3",
#                 save=True, name="sc_pca3_selective")
# k = tsne_cluster(X=x, Y=y, components=2, visualize=True, iterations=10000,
#                   tit="TSNE-2", save=True, name="sc_tsne2_selective")
# k = tsne_cluster(X=x, Y=y, components=3, visualize=True, iterations=10000,
#                   tit="TSNE-3", save=True, name="sc_tsne3_selective")


# # Deep classification parts (colab recommended for this part)

# dataset2 = SampleDataset()
# dataset2.X = x[3:16, :]
# dataset2.Y = y[3:16]
# classifier1 = DeepModel2(input_size=in_shape, dataset=dataset2)

# classifier1.train(200, 10, val=0.2)
# print(classifier1.encoder(x), "\n", y)




# # Graph visualize saving frames
# run_graph_visualize(ds_spectral, mode="Spectral", split="count-quality")
# run_graph_visualize(ds_pearson, mode="Spectral", split="count-quality")


# Animator
# make_gif(path="Graphs/Spectral_Good_count-quality_90_f_bandpass[15_25]Hz/",
#           fname="Animate/sg90_15_25.gif", duration=0.23, f=70)
# make_gif(path="Graphs/Spectral_Bad_count-quality_90_f_bandpass[15_25]Hz/",
#           fname="Animate/sb90_15_25.gif", duration=0.23, f=70)
# make_gif(path="Graphs/Spectral_Good_count-quality_10_f_bandpass[32_38]Hz/",
#           fname="Animate/sg10_32_38.gif", duration=0.12)
