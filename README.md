# Mental Arithmetics EEG Coherence Graph

### Analysis of cortical connectivity based on EEG recordings during mental arithmetics

B.sc thesis of Hamed Nejat, EE/Bio-electrics @ SUT (2021)

## Connectivity

In a system like human brain, the similarity between recorded signals (electromagnetic, chemical, etc...) can give us a good information about functions of brain and later investigations about related diseases. The similarity criterion and variable normalizations finally form a weight matrix that can be transformed into a weighted graph, and if criterion is causal, a directional graph. Due to lower SNR than some other type of signals, we did not apply causal methods (Granger, Phase, etc...) on this dataset.

### Local connectivity effects

Due to electromagnetic properties of cortical matter, closer areas have more coherence; this problem causes the graph to be strongly connected in local nodes and weaker in distance nodes. In order to reduce this effect, we measured pink noise effect versus distance and normalized graph weights based on different frequency amplitude changes across the whole network. 

#### 30 frame (~2sec) window spectral coherence based graphs:

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sg30_15_25.gif" width="400"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sb30_15_25.gif" width="400"/>

#### 10 frame (~6sec) window mutual information based graphs:

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sg10_32_38.gif" width="400"/>

## Connectivity graph weight statistical test

In order to determine if some edges are important or not, we performed a statistical t-test on graph edges, based on two defined labels; high performance (good) calculator subject's EEG signal and low performance (bad) subject's EEG. Its resault are shown below:

#### P-values(right) and stats(left)

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/ztpval_fs_0_ff_1.png" width="400"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/ztstat_fs_0_ff_1.png" width="400"/> 

So it means, the coherence leading to [F8, O2, P3] electrodes are different for two set of subjects on average. This became validated via training a deep feedforward classifier for two sets. The dataset was limited and small (22 subjects) so an augmentation applied on data via increasing frames to 5 and doing a shuffle on new 110 (22x5) data (just for classification). Red points are high performance subject's and blue are for low. Consider that there is no guarantee that EEGMAT-Physionet2018-MentArth dataset labeling (performance and number of actions per minute) are so accurate, as there were some subjects with 1 action in 60 seconds, so far from average subjects with more than 15 action per 60 seconds.


## Unsupervised results

For better illustration, we calculated unsupervised TSNE/PCA clustering results on weighted graphs.

### Spectral coherence based:

The next 4 plots are based on spectral coherence graphs:
 
#### PCA(2D/3D)

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_pca2.png" width="492"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_pca3.png" width="492"/>

#### TSNE(2D/3D)

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_tsne2.png" width="492"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_tsne3.png" width="492"/>

### Mutual information based:
Mutual information based graphs are calculated based on MI value between spectrum of two signals, the next 4 plots:

#### PCA(2D/3D)

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_pca2.png" width="492"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_pca3.png" width="492"/>

#### TSNE(2D/3D)

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_tsne2.png" width="492"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_tsne3.png" width="492"/>


On MI (mutual information) based plots we can draw eclipses that separate two label classes with more than 85% accuracy (TSNE) and 80% (PCA). Yet on the SC (spectral coherence) based ones this value is respectively 75% and 70%. SC based values are still plausible but MI based ones are much better, showing the better representation in distribution domain of signals.

Difference between TSNE and PCA also shows the non-linear pattern extraction advantage in such task. Consider that these EEG recordings are based on standard 10-20, non invasive and we expect much more accuracy and representation when these methods are applied on sEEG or even ECoG with much higher SNR. 

### Selective graph weight results

In order to enhance clustering and classification models, set of about 20-30 most significant electrodes based on T-Test p-values got selected. Following plots are the same clusteing (PCA/TSNE) repeated with these electrodes, enhancing clustering results:


#### PCA(2D/3D)

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_pca2_selective.png" width="492"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_pca3_selective.png" width="492"/>

#### TSNE(2D/3D)

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_tsne2_selective.png" width="492"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/sc_tsne3_selective.png" width="492"/>

### Mutual information based:
Mutual information based graphs are calculated based on MI value between spectrum of two signals, the next 4 plots:

#### PCA(2D/3D)

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_pca2_selective.png" width="492"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_pca3_selective.png" width="492"/>

#### TSNE(2D/3D)

<img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_tsne2_selective.png" width="492"/> <img src="https://github.com/HNXJ/MentalArthGraph/blob/main/Plots/mi_tsne3_selective.png" width="492"/>

### Comparison

Clustering results based on MI method are better spearated related to SC method; although two of data nodes in the clustering are not well separated. This can be due to very different mental state of those subjects. for example, one may not focus on the tasks and still do the calculations in a flausible result. Even with assuming those data as error of our clustering and classifiers the test accuracy is more than 84%.

### Overall results

* EEG-15-25Hz frequency band, SC and MI on 3s time windows (20 frame)
* Most significant channels: [F8, P3, O1, F7, T3, T4]
* Most significant edges: [F8-F7,F8-T3,F8-P3,T3-P3,O1-T3,T4-T3]
* Graph weights average std per subject is more on subjects with good calculation qualities; that means total variation of a good subject's graph is higher than a bad one 
* Electrode distance has less effect on MI graphs than SC
* Subjects with higher performance have more coherence between F8 signal and 6 other electrodes near occipital and temporal area.
* Current SNR was low for event based pattern recognition (low significance)


