# Bi-TGCF
Tensorflow Implementation of BiTGCF：
Cross Domain Recommendation via Bi-directional Transfer Graph Collaborative Filtering Networks. in CIKM2020

## DATASET
Download .csv file (rating only) from [Amazon](http://jmcauley.ucsd.edu/data/amazon/). The information about elec and Cell datasets in the paper were wrong. The first pair of datasets used in my paper is given here. The rest of the datasets are correct.

## RUN
    python main.py --dataset=elctronic_cell --n_interaction=3 --lambda_s=0.8 --lambda_t=0.8
