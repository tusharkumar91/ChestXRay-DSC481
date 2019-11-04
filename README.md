# ChestXRay-DSC481
Repository for Predicting Pneumonia using Chest XRay images. Done as part of the final Project for DSC481 course at University of Rochester.

## Setup
```
Create data folder in the repo
Download image folders from https://kaggle.com/paultimothymooney/chest-xray-pneumonia/
Unzip downladed folder and place it in data directory
Verify the structure of data directory to be the following
1. data
   - chest_xray
     - train
     - val
     - test
```
## Training
Run the following command from the root directory
`python -m scripts.train`
