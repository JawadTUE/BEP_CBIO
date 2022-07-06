# BEP_CBIO

## CODE
### main.ipynb
In this file, the overlap function is called, the similarity matrix is computed and the HCS clustering is performed.

### data_preprocessing.ipynb
In this file, the .bed files are converted to the csv file.

### Overlap.py
This file contains the overlap algorithms. 
The n2_overlap() function is just the brute force method used in testing.
The remaining functions are part of the sweep line algorithm.

### Graph.py
Class to create a graph object. 

### HCS.py
Highly connected subgraph class.

## DATA
### region_data.csv and similarity_matrix.npy are to big for github
### metadata.csv
The first line shows the tissues in the dataset (in correct order)
The second line shows the amount of active regions in the tissues of line 1
The last line shows the chromosomes (and in which order) used in this work

### overlap_matrix.npy
Amount of overlap between the tissues in a numpy matrix



