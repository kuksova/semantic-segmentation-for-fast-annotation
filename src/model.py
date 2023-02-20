from PIL import Image
import torch
import torch.nn as nn
import numpy as np


from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

from src.evaluate_sample import evaluate_sam, load_sample
from src.convert_pred_mask import get_blind, overlay
from src.mit_seg_transform import imresize


def base_model(num_classes):
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights, progress=False)

    #print(model.classifier)
    #print('Last layer of section is \n', model.classifier[4])

    # num_classes = 3 #num classes of my dataset

    filters_of_last_layer = model.classifier[4].in_channels
    filters_of_last_layer_aux = model.aux_classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(filters_of_last_layer, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(filters_of_last_layer_aux, num_classes, kernel_size=(1, 1), stride=(1, 1))
    # Now our model is finetuned for my dataset
    #print(model.classifier)

    return model

def get_separate_mask(mask):
    classes = np.unique(mask)

    lst_all = []
    for i in classes:
        a = mask == i
        lst_all.append(a.astype(np.uint8)*255)

    return lst_all[1:]

def predict(image_path):

    # classes = {1 : "wall",
    #          2 : "floor",
    #          3: "sky"}

    output_classes = 4  # 3 main classes + class "background"
    model = base_model(output_classes)  # already fine-tuned

    device = torch.device('cpu')
    PATH = '../model1.pth'
    
    model.load_state_dict(torch.load(PATH, map_location=device))

    img, w_orig, h_orig = load_sample(image_path)

    result = evaluate_sam(img, model, device)

    # IOU metric ??
    # probability??
    # show ??

    result = result.numpy()
    #result_all = np.array(result)
    result_all = get_separate_mask(result)

    img = img.numpy()
    #uint_mask = get_blind(img, result)
    #pil_mask = Image.fromarray(uint_mask)
    pil_image = Image.open(image_path)
    pil_mask = overlay(pil_image, result, output_classes)
    pil_mask = imresize(pil_mask, (w_orig, h_orig), interp='nearest')
    
    


    #pil_mask = Image.fromarray(uint_mask)# Get Pil Image? done.


    return pil_mask, result_all
