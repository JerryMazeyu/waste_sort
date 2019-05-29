import torch as t
from torchvision.datasets import ImageFolder
import torchvision as tv
from torchvision import transforms as T
from config import opt
from torchvision import models
from torch import nn



# model
resnet34 = models.resnet34(pretrained=True, num_classes=1000)
resnet34.fc = nn.Linear(512, 6)
device = t.device('cuda:0' if opt.use_gpu else 'cpu')
resnet34.to(device)


# load data
transforms = tv.transforms.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = ImageFolder(opt.data_path, transform=transforms)
print("there is %s in this dataset." % len(dataset.imgs))
train_db, val_db = t.utils.data.random_split(dataset, [25, 2502])
print('train:', len(train_db), 'validation:', len(val_db))
dataloader_train = t.utils.data.DataLoader(train_db, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)
dataloader_val = t.utils.data.DataLoader(val_db, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)



# critrion and loss_fun
criterion = nn.CrossEntropyLoss()
lr = opt.lr
optimizer = t.optim.Adam(resnet34.parameters(), lr=lr, weight_decay=opt.weight_decay)



# train
resnet34_finetune_wt = resnet34.state_dict()
for epoch in range(opt.max_epoch):
    print("epoch is: ", epoch)

    for ii, (input, target) in enumerate(dataloader_train):
        print("ii is: ", ii)
        input.to(device)
        target.to(device)

        optimizer.zero_grad()
        score = resnet34(input)
        loss = criterion(score, target)
        loss.backward()
        optimizer.step()
        print("loss is: ", loss)

t.save(resnet34_finetune_wt, opt.net_path)
