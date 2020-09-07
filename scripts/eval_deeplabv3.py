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
num_classes = 1
classes_map = {"0": (0, 0, 0), "1": (255, 255, 255)}

# define transforms
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
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
    output = output.squeeze().detach().cpu().numpy()
    
    # get image from output
    r = np.zeros_like(output).astype(np.uint8)
    g = np.zeros_like(output).astype(np.uint8)
    b = np.zeros_like(output).astype(np.uint8)
    for index1 in range(0, output.shape[0]):
        for index2 in range(0, output.shape[1]):
            if(output[index1, index2] > 0.2):
                r[index1, index2] = 255
                g[index1, index2] = 255
                b[index1, index2] = 255
            else:
                r[index1, index2] = 0
                g[index1, index2] = 0
                b[index1, index2] = 0
    output = np.stack([r, g, b], axis=2)
    _ = Image.fromarray(output).save(output_files_path + files[index].split("/")[-1])
