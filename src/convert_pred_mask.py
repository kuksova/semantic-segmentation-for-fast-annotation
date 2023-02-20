import numpy as np
from PIL import Image, ImageDraw

def get_palette():
    palette = np.array([[0,   0,   0],
       [8,   184,   170 ],
       [  255, 224,   0],
       [6, 230,  230]], dtype=np.int32)
    return palette

def visualize_mask(mask):
    palette = get_palette()
    return palette[mask].astype(np.uint8)

def get_blind(img, mask):
  img = np.transpose(img, (1,2,0))
  alpha=0.4
  out = img.copy()

  palette = get_palette()
  rgb_mask = palette[mask.astype(np.uint8)]

  #rgb_mask = rgb_mask.astype(np.uint8)
  #print(rgb_mask.shape)
  #print(rgb_mask)

  mask_region = (mask > 0).astype(np.uint8)
  out = out * (1 - mask_region[:, :, np.newaxis]) + \
      (1 - alpha) * mask_region[:, :, np.newaxis] * out + \
      alpha * rgb_mask
  out = out.astype(np.uint8)
  return out

def overlay(image, mask, num_classes):
    #img = np.transpose(img, (1,2,0))
    #image = Image.fromarray(img.astype('uint8'), 'RGB').convert('RGBA')
    image.putalpha(255)
    
    
    overlay = Image.new('RGBA', image.size, (255,255,255,0))
    drawing = ImageDraw.Draw(overlay)
    
    for i in range(1, num_classes+1):
        current_mask = (mask == i).astype(np.uint8) * 255
        current_mask = Image.fromarray(current_mask, mode='L')
        current_mask = current_mask.resize(image.size, Image.NEAREST)
        drawing.bitmap((0, 0), current_mask, fill=(255*(i==1), 255*(i==2), 255*(i==3), 128))
        
    image = Image.alpha_composite(image, overlay)
    return image

    
    