## This is a pytorch implementation of the paper *[Domain Adaptation based Unsupervised Cross-Type Deepfake Image Detection]*


#### Environment
- Pytorch 1.12
- Python 3.7
  
#### Requirement
-dlib 19.22.1
-numpy 1.21.6
-opencv-python 4.5.5.62
-pandas 1.2.3
-scipy 1.7.3
-torch 1.12.1
-scikit-image 0.19.1 

#### Network Structure

![绘图36](https://github.com/user-attachments/assets/e2d2dc81-5063-4914-b5d3-d438999c5fc5)


#### Dataset
# Validation data
source_dataset_name = './Dataset/source/test' 
target_dataset_name = './Dataset/target/test'

# Train data
source_image_root = './Dataset/source/train'
target_image_root = './Dataset/source/train'

​Place the directory in which each set of data resides within double quotes in the code above. Note that the file directory for each set of data contains two subdirectories, Real and Fake, which contain their respective images. 

```

#### Training

Then, run `main.py`


#### Test

 run `test_classifier.py`


