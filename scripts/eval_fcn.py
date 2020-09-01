# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
import glob


# constants
model_path = "/home/arpitdec5/Desktop/cnn_architectures_image_segmentation/scripts/best_model.pth"
files = "/home/arpitdec5/Desktop/cnn_architectures_image_segmentation/data/"
files = glob.glob(files + "/*")
output_files_path = "/home/arpitdec5/Desktop/cnn_architectures_image_segmentation/output/"
num_classes = 21
classes_map = {"0": (0, 0, 0), "1": (128, 0, 0), "2": (0, 128, 0), "3": (128, 128, 0), "4": (0, 0, 128), "5": (128, 0, 128), "6": (0, 128, 128), "7": (128, 128, 128), "8": (64, 0, 0), "9": (192, 0, 0), "10": (192, 128, 0), "11": (64, 0, 128), "12": (192, 0, 128), "13": (64, 128, 128), "14": (192, 128, 128), "15": (0, 64, 0), "16": (128, 64, 0), "17": (0, 192, 0), "18": (128, 192, 0), "19": (0, 64, 128), "20": (64, 128, 0)}

# define transforms
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                       torchvision.transforms.CenterCrop((224, 224)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
model.classifier = torchvision.models.segmentation.fcn.FCNHead(2048, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# eval model
model.eval()
for index in range(0, len(files)):
    # read image
    image = Image.open(files[index])

    # run model on image
    input = transforms(image).unsqueeze(0)
    output = model(input)['out']
    output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    
    # get image from output
    r = np.zeros_like(output).astype(np.uint8)
    g = np.zeros_like(output).astype(np.uint8)
    b = np.zeros_like(output).astype(np.uint8)
    for index1 in range(0, output.shape[0]):
        for index2 in range(0, output.shape[1]):
            r[index1, index2] = classes_map[str(output[index1, index2])][0]
            g[index1, index2] = classes_map[str(output[index1, index2])][1]
            b[index1, index2] = classes_map[str(output[index1, index2])][2]
    output = np.stack([r, g, b], axis=2)
    _ = Image.fromarray(output).save(output_files_path + files[index].split("/")[-1])
