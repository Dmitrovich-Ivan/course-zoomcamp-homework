### INTRODUCTION:

Oil spills can be defined as the release of liquid raw/natural petroleum hydrocarbons into the environment.  Not only oil spills pose a great threat to all forms of life and ecosystems located near them, but damage different spheres of economic activity. At the same time, existing cleaning methods of such pollutions are time and resource-consuming, require coordination and depend on various external conditions. All of that makes the task of identifying and locating the oil spills even more relevant. 

### DATA:

The dataset can be downloaded [here](https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection). 
The dataset was developed by collecting satellite images of the ocean which were split into sections and processed using computer vision algorithms to provide a vector of features to describe the contents of the image section.

### OBJECTIVE:

The task is, given a vector that describes the contents of a patch of a satellite image, predict whether it contains an oil spill or not. There are two classes and the goal is to distinguish between spill and non-spill using the features for a given ocean patch. 
- Non-Spill: negative case (0). 
- Oil Spill: positive case (1).

### CONTENTS:
This repository contains the following files:
   - `notebook.ipynb` with data preparation, EDA, model training and selection process.
   - `train.py` - code for training and saving models. 
   - `oil_spill.csv` - the dataset that was used for this project. It can be downloaded from this repository as well.

### MODEL:
   Various models and approaches were tested and XGBoost performed better than others.

