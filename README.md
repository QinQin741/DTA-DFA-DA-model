## This is a pytorch implementation of the paper *[Domain Adaptation based Unsupervised Cross-Type Deepfake Image Detection]*


#### Environment
- Pytorch 1.12
- Python 3.7

#### Network Structure


![p8KTyD.md.jpg](https://s1.ax1x.com/2018/01/12/p8KTyD.md.jpg)

#### Dataset
# Test data
source_dataset_name = './Dataset/source/test' 
target_dataset_name = './Dataset/target/test'

# Train data
source_image_root = './Dataset/source/train'
target_image_root = './Dataset/source/train'

​ Place the directory in which each set of data resides within double quotes in the code above. Note that the file directory for each set of data contains two subdirectories, Real and Fake, which contain their respective images. 

```

#### Training

Then, run `python main.py`


#### Docker

- build image

```bash
docker build -t pytorch_dann .
```

- run docker container

```bash
docker run -it --runtime=nvidia \
  -u $(id -u):$(id -g) \
  -v /YOUR/DANN/PROJECT/dataset:/DANN/dataset \
  -v /YOUR/DANN/PROJECT/models:/DANN/models \
  pytorch_dann:latest \
  python main.py

```

