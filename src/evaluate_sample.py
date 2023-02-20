from src.mit_seg_transform import imresize, img_transform, segm_transform

# Load sample and mask 
#some test
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_sample(img_name):
  w = 224
  h = 224
  
  image = Image.open(img_name).convert("RGB")
  w1, h1 = image.size

  #plt.imshow(np.array(image))  # display image
  #plt.show()

  image = imresize(image, (w, h), interp='bilinear')

  # image transform, to torch float tensor 3xHxW
  img = img_transform(image)
  return img, w1, h1

def evaluate_sam(img, model, device):
  img = img.unsqueeze(0)
  if device == torch.device('cuda'):
    img = img.to(device)
    model = model.to(device)
  model.eval()
  output_pred = model(img)['out']
  print(output_pred.shape, output_pred.min().item(), output_pred.max().item())
  output_pred.size()

  normalized_masks_pred = torch.nn.functional.softmax(output_pred, dim=1)[0]
  #print(normalized_masks_pred.size())

  #prd2=model(img2)['out'][0]
  if device == torch.device('cuda'):
    result = torch.argmax(normalized_masks_pred, 0).cpu().detach() 
  else:
    result = torch.argmax(normalized_masks_pred, 0).detach()

  return result

