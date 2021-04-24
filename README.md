# MentalArthGraph

### Analysis of cortical connectivity based on EEG recordings during mental arithmetics

B.sc thesis of HN, SUT (2021)

## Connectivity

In a system like human brain, the similarity between recorded signals (electromagnetic, chemical, etc...) can give us a good information about functions of brain and later investigations about related diseases. The similarity criterion and variable normalizations finally form a weight matrix that can be transformed into a weighted graph, and if criterion is causal, a directional graph. 

#### 12 frame (5sec) window spectral coherence based graphs:
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sg12_60_120.gif)
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sb12_60_120.gif)

#### 10 frame (6sec) window mutual information based graphs:
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/sg10_32_38.gif)

## Connectivity graph edge weight test

In order to determine if some edges are important or not, we performed a statistical t-test on graph edges, based on two defined labels; high performance (good) calculator subject's EEG signal and low performance (bad) subject's EEG. Its resault are shown below:

#### Stats
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/ztpval_fs_0_ff_1.png)
#### P-values
![Alt Text](https://github.com/HNXJ/MentalArthGraph/blob/main/Animate/ztpval_fs_0_ff_1.png)

So it means, the coherence leading to [F8, O1, P3] electrodes are on average different for two set of subjects.
