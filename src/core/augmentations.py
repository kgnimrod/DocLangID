import math
import torch
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def generate_op_meta(numb_magnitudes, image_size):
    return {
        # op_name: (magnitudes, signed)
        "Identity": (torch.tensor(0.0), False),
        "ShearX": (torch.linspace(0.0, 0.2, numb_magnitudes), True),
        "ShearY": (torch.linspace(0.0, 0.2, numb_magnitudes), True),
        "Rotate": (torch.linspace(0.0, 45.0, numb_magnitudes), True),
        "Brightness": (torch.linspace(0.0, 0.7, numb_magnitudes), True),
        "Contrast": (torch.linspace(0.0, 0.7, numb_magnitudes), True),
        "Sharpness": (torch.linspace(0.0, 0.9, numb_magnitudes), True),
        "Equalize": (torch.tensor(0.0), False), # adds noise
    }

def randaugment(image, ops, mag, numb_magnitudes, interpolation=InterpolationMode.BILINEAR, fill=[255, 255, 255]):
    # get meta information
    #op_meta = generate_op_meta(numb_magnitudes, (image.shape[1], image.shape[2]))
    op_meta = generate_op_meta(numb_magnitudes, (image.height, image.width))
    indexes = list(range(len(op_meta)))
    indexes = random.sample(indexes, k=ops[0])
    for i in indexes:
        op_name = list(op_meta.keys())[i]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[mag[0]].item()) if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        image = apply_op(image, op_name, magnitude, interpolation, fill)
    return image

def apply_op(image, op_name, magnitude, interpolation, fill):
    if op_name == 'ShearX':
        image = TF.affine(
            image,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == 'ShearY':
        image = TF.affine(
            image,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        image = TF.affine(
            image,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        image = TF.affine(
            image,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        image = TF.rotate(image, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        image = TF.adjust_brightness(image, 1.0 + magnitude)
    elif op_name == "Color":
        image = TF.adjust_saturation(image, 1.0 + magnitude)
    elif op_name == "Contrast":
        image = TF.adjust_contrast(image, 1.0 + magnitude)
    elif op_name == "Sharpness":
        image = TF.adjust_sharpness(image, 1.0 + magnitude)
    elif op_name == "Solarize":
        image = TF.solarize(image, magnitude)
    elif op_name == "Equalize":
        #image = TF.equalize(image.to(torch.uint8))
        #image = image.to(torch.float32)
        image = TF.equalize(image)
    elif op_name == "Posterize":
        image = TF.posterize(image, int(magnitude))
    elif op_name == "AutoContrast":
        image = TF.autocontrast(image)
    elif op_name == "Invert":
        image = TF.invert(image)
    elif op_name == "HFlip":
        image = TF.hflip(image)
    elif op_name == "VFlip":
        image = TF.vflip(image)
    elif op_name == "Identity":
        pass
    else:
        print(f"Unknown op: {op_name}")
        pass
    
    return image