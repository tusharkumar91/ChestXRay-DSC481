Runtime Instructions
--------------------
Github Link - https://github.com/tusharkumar91/ChestXRay-DSC481

1. Download dataset from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
2. Place the chest_xray folder within data folder 
3. Train the resent model using 'python -m scripts.train'
4. Test the model using 'python -m scripts.test'


File Descriptions
-----------------
chest_xray_data_loader - Data loader for our chest x-ray dataset 
data - Folder comprising the json files with original Kaggle and our 70-15-15 split.
explain.py - Visualization file taken from https://github.com/jacobgil/pytorch-explain-black-box
models - Folder containing our model files 
	UNet files : models/build_model.py, models/unet_parts.py from https://github.com/MEDAL-IITB/Lung-Segmentation/tree/master/VGG_UNet/code
	Resent Model - models/resnet_xray_model.py
	VGG Model - present in script/train.py initialized during training itself and does not have a separate class for it.
README.md - Github Readme file
utils - Folder comprising our utilities files
	augmentation.py - Augmentation script used in our data loader
	metric.py - Accuracy, Precision and Recall calculation code
scripts - Folder containing training and testing scripts 
	train.py - Training script initializing and training Resnet 34 model and saving checkpoints
	test.py - Loads saved resnet checkpoint and evaluates on testing set.

