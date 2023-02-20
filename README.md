# semantic-segmentation-for-fast-annotation

The project aims to simplify the automatic annotation process of indoor and outdoor images by fine-tuning a general segmentation model to pre-selected popular objects (e.g. sky, floor, tree, etc). 

This approach limits a variety of classes to be considered and therefore significantly reduces complexity of delivered segmentation masks.


![Screenshot from 2023-02-20 13-36-44](https://user-images.githubusercontent.com/14224692/220201863-7b5d6223-3e40-4b60-a23e-78b676b7e915.png)

## Setting up an environment
This framework is built using Python 3.8 and PyTorch. The following command installs all necessary packages:
```
pip3 install -r requirements.txt
```

## Demo 
The model is based on FCN_ResNet50, trained on 3 pre-selected classes. The model was trained on GPU. You need to dowload it and add to the project's root. The link with the model is  
https://drive.google.com/file/d/1BYUMuyaRxU3RajqRlkEL0V7uFCa4fz7t/view?usp=sharing 

## Usage 
Run main.py 

```
$python main.py 
```

## Presentation
You can check the presentation about this project.
https://docs.google.com/presentation/d/1uC3RYIR-Io9yPohW44WQeZ9Knqct4mgR/edit?usp=sharing&ouid=103552389649049130550&rtpof=true&sd=true
