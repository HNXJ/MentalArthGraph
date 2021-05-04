# Mental Arithmetics EEG Coherence Graph

### Analysis of cortical connectivity based on EEG recordings during mental arithmetics

B.sc thesis of Hamed Nejat, EE/Bio-electics @ SUT (2021)

## Connectivity

In a system like human brain, the similarity between recorded signals (electromagnetic, chemical, etc...) can give us a good information about functions of brain and later investigations about related diseases. The similarity criterion and variable normalizations finally form a weight matrix that can be transformed into a weighted graph, and if criterion is causal, a directional graph. Due to lower SNR than some other type of signals, we did not apply causal methods (Granger, Phase, etc...) on this dataset.

### Local connectivity effects

Due to electromagnetic properties of cortical matter, closer areas have more coherence; this problem causes the graph to be strongly connected in local nodes and weaker in distance nodes. In order to reduce this effect, we measured pink noise effect versus distance and normalized graph weights based on different frequency amplitude changes across the whole network. 

#### 30 frame (~2sec) window spectral coherence based graphs:
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sg30_15_25.gif)
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sb30_15_25.gif)

#### 10 frame (~6sec) window mutual information based graphs:
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sg10_32_38.gif)

## Connectivity graph edge weight test

In order to determine if some edges are important or not, we performed a statistical t-test on graph edges, based on two defined labels; high performance (good) calculator subject's EEG signal and low performance (bad) subject's EEG. Its resault are shown below:


#### P-values and stats
![alt-text-1](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/ztpval_fs_0_ff_1.png "Pvalues") ![alt-text-2](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/ztstat_fs_0_ff_1.png "Stats")

%<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/ztpval_fs_0_ff_1.png" width="425"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/ztstat_fs_0_ff_1.png" width="425"/> 

So it means, the coherence leading to [F8, O2, P3] electrodes are different for two set of subjects on average. This became validated via training a deep feedforward classifier for two sets. The dataset was limited and small (22 subjects) so an augmentation applied on data via increasing frames to 5 and doing a shuffle on new 110 (22x5) data (just for classification). Red points are high performance subject's and blue are for low. Consider that there is no guarantee that EEGMAT-Physionet2018-MentArth dataset labeling (performance and number of actions per minute) are so accurate, as there were some subjects with 1 action in 60 seconds, so far from average subjects with more than 15 action per 60 seconds.


## Unsupervised results

For better illustration, we calculated unsupervised TSNE/PCA clustering results on weighted graphs.

### Spectral coherence based:
The next 4 plots are based on spectral coherence graphs:
 
#### PCA-2D :
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_pca2.png)

#### PCA-3D
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_pca3.png)

#### TSNE-2D
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_tsne2.png)

#### TSNE-3D
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_tsne3.png)


### Mutual information based:
Mutual information based graphs are calculated based on MI value between spectrum of two signals, the next 4 plots:

#### PCA-2D :
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_pca2.png)

#### PCA-3D
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_pca3.png)

#### TSNE-2D
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_tsne2.png)

#### TSNE-3D
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_tsne3.png)


On MI (mutual information) based plots we can draw eclipses that separate two label classes with more than 85% accuracy (TSNE) and 80% (PCA). Yet on the SC (spectral coherence) based ones this value is respectively 75% and 70%. SC based values are still plausible but MI based ones are much better, showing the better representation in distribution domain of signals.

Difference between TSNE and PCA also shows the non-linear pattern extraction advantage in such task. Consider that these EEG recordings are based on standard 10-20, non invasive and we expect much more accuracy and representation when these methods are applied on sEEG or even ECoG with much higher SNR. 

### Selective graph weight results

In order to enhance clustering and classification models, set of about 20-30 most significant electrodes based on T-Test p-values got selected. Following plots are the same clusteing (PCA/TSNE) repeated with these electrodes, enhancing clustering results:

(/TODO PCA/TSNE on 20-30 selectvie edges with lowest P-value on TTest/)
