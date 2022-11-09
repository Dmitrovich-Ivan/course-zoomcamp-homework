### INTRODUCTION: 
   Medical insurance premium is the amount of money paid to the insurance provider to get health insurance coverage. The calculation of the medical insurance premiums is a non trivial task since it depends on a various set of factors. It requires a comprehensive analysis of the customer profile, historical data as well as assessing the risk factors. It is important to know that every insurer uses its own assumptions and has its own set of standards to calculate the health insurance premium. 
   
### DATA: 
   The dataset can be downloaded [here](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction).
   It contains health related parameters of the customers. The premium price is in inr(₹) currency and shows annual amounts. 
   
### OBJECTIVE: 
   Use machine learning approach in order to simplify the process of calculating the premium amount as for customers as for insurance providers at the same time providing accurate, reliable results and additional insights. 
   
### CONTENTS:
   This repository contains the following files:
   - `notebook.ipynb` with data preparation, EDA, model training and selection process.
   - `train.py` - code for training and saving models. 
   - `service.py` which loads the saved model and contains code for serving it using BentoML. 
   - `bentofile.yaml` with dependencies, package versions, descriptions and some other info required for building bentos with BentoML. 
   - `medicalpremium.csv` - the dataset that was used for this project. It can be downloaded from this repository as well. 
   
### HOW TO RUN THE MODEL:
   The easiest way to run and test the model is to pull the docker image of that model from this [public repository](https://hub.docker.com/r/vanyadmitrovich/premium-prediction). Run this command in order to download the image:
   - `docker pull vanyadmitrovich/premium-prediction`
   
  Use the following command to run the image:
   - `docker run -it --rm -p 3000:3000 vanyadmitrovich/premium-prediction:latest serve --production`

  After that the service will be running on http://localhost:3000/ 
 
 
