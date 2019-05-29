import take_photo
import torch as t
from torchvision.datasets import ImageFolder
import torchvision as tv
from torchvision import transforms as T
from torchvision import models
from torch import nn
from config import opt
import tkinter as tk



def output_label(input_path='./cache/', state_dict='./checkpoints/resnet34_finetune_wt.pth', id2class=opt.id2class, verbose=True):
    transforms = tv.transforms.Compose([
        T.Resize(384),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(input_path, transform=transforms)
    input = dataset[0][0]
    print(input.size())
    input = t.unsqueeze(input, dim=0)
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(512, 6)
    model.load_state_dict(t.load(state_dict))
    output = model(input)
    output_index = t.max(output, 1)[1].numpy()[0]
    output_label = list(id2class.keys())[list(id2class.values()).index(output_index)]
    if verbose:
        print("label is: ", output_label)
    return output



if __name__ == '__main__':
    take_photo.take_photo()
    output_label()