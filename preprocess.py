import numpy as np
from keras.utils import to_categorical


def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]


def normalize_2D_image(img):
    v_max = img.max(axis=(0, 1), keepdims=True)
    v_max = np.where(np.abs(v_max) <= 0.000001, 1., v_max)

    return (img) / (v_max)


def normalize_3D_image(img):
    for z in range(img.shape[0]):
        img[z] = normalize_2D_image(img[z])
    return img

# Return label to categorical
def label_num2cat(label,num_class):
  new_mask = np.zeros(label.shape + (num_class,))
  mask = to_categorical(label,5)
  for i, c in enumerate([0,1,2,4]):
    new_mask[:,:,:,i] = mask[:,:,:,c]
  return new_mask


# Return label from categorical
def label_cat2num(cat_lbl):
    lbl = 0
    if (len(cat_lbl.shape) == 3):
        for i in range(1, 4):
            lbl = lbl + cat_lbl[:, :, i] * i
    elif (len(cat_lbl.shape) == 4):
        for i in range(1, 4):
            lbl = lbl + cat_lbl[:, :, :, i] * i
    else:
        print('Error in lbl_from_cat', cat_lbl.shape)
        return None
    return lbl

def preprocess(img, mask):
    img = normalize_3D_image(img)
    print(img.shape)
    new_mask = label_num2cat(mask)
    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(img)
    return img[:,cmin:cmax], new_mask[cmin:cmax]