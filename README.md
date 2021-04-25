# MentalArthGraph

### Analysis of cortical connectivity based on EEG recordings during mental arithmetics

B.sc thesis of Hamed Nejat, EE/Bio-electics @ SUT (2021)

## Connectivity

In a system like human brain, the similarity between recorded signals (electromagnetic, chemical, etc...) can give us a good information about functions of brain and later investigations about related diseases. The similarity criterion and variable normalizations finally form a weight matrix that can be transformed into a weighted graph, and if criterion is causal, a directional graph. 

#### 30 frame (5sec) window spectral coherence based graphs:
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sg30_15_25.gif)
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sb30_15_25.gif)

#### 10 frame (6sec) window mutual information based graphs:
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sg10_32_38.gif)

## Connectivity graph edge weight test

In order to determine if some edges are important or not, we performed a statistical t-test on graph edges, based on two defined labels; high performance (good) calculator subject's EEG signal and low performance (bad) subject's EEG. Its resault are shown below:


#### P-values
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/ztpval_fs_0_ff_1.png)

#### Stats
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/ztstat_fs_0_ff_1.png)

So it means, the coherence leading to [F8, O2, P3] electrodes are different for two set of subjects on average. This became validated via training a deep feedforward classifier for two sets. The dataset was limited and small (22 subjects) so an augmentation applied on data via increasing frames to 5 and doing a shuffle on new 110 (22x5) data (just for classification). 

## Unsupervised results

For better illustration, we calculated unsupervised TSNE/PCA clustering results on weighted graphs:
