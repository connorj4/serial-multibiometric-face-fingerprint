import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
class SiameseNetworkFinger(nn.Module):
    def __init__(self):
        super(SiameseNetworkFinger, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 64, kernel_size=8),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),
            nn.BatchNorm2d(128),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),
            nn.BatchNorm2d(256),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),
            nn.BatchNorm2d(512),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(5*5*512, 800),
            nn.ReLU(inplace=True),

            nn.Linear(800, 1000),
            nn.ReLU(inplace=True),

            nn.Linear(1000, 800),
            nn.ReLU(inplace=True),

            nn.Linear(800, 40), #3
            nn.Sigmoid(),
            )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class SiameseNetworkFace(nn.Module):
    def __init__(self):
        super(SiameseNetworkFace, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 64, kernel_size=8),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),
            nn.BatchNorm2d(128),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),
            nn.BatchNorm2d(256),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),
            nn.BatchNorm2d(512),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(5*5*512, 800),
            nn.ReLU(inplace=True),

            nn.Linear(800, 1000),
            nn.ReLU(inplace=True),

            nn.Linear(1000, 800),
            nn.ReLU(inplace=True),

            nn.Linear(800, 40), #3
            nn.Sigmoid()
            )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

face_path_1 = './images/6.pgm'
face_path_2 = './images/9.pgm'
finger_1 = './images/r42.jpg'
finger_2 = './images/r39.jpg'

img0 = Image.open(face_path_1)
img1 = Image.open(face_path_2)
plt.imshow(img0)
plt.show()
plt.imshow(img1)
plt.show()
transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
img0=transform(img0)
img1=transform(img1)

img0 = img0.view(-1,1,100,100) #expecting an extra dimension (bc technically minibatch of 1)
img1 = img1.view(-1,1,100,100)

net = SiameseNetworkFace()
sd = torch.load('./weights/InfoSecFace.pth',map_location='cpu') #we want to do everything on the CPU! (if deploying)
net.load_state_dict(sd)
net.eval()
output1,output2 = net(img0,img1)
euclidean_distance = F.pairwise_distance(output1, output2)
sim = 1-euclidean_distance.item()/2
if(sim<.25):
  print(f'Not Accepted with similarity score of {sim}')
elif(sim>=.25 and sim<.8):
  print(f'Similarity score is {sim} please give fingerprint')
  input()
  img2 = Image.open(finger_1)
  img3 = Image.open(finger_2)
  plt.imshow(img2)
  plt.show()
  plt.imshow(img3)
  plt.show()
  img2=transform(img2)
  img3=transform(img3)
  img2 = img2.view(-1,1,100,100) #expecting an extra dimension (bc technically minibatch of 1)
  img3 = img3.view(-1,1,100,100)
  net = SiameseNetworkFinger()
  sd = torch.load('./weights/InfoSecFingerprint.pth',map_location='cpu')
  net.load_state_dict(sd)
  net.eval()
  output1,output2 = net(img2,img3)
  euclidean_distance = F.pairwise_distance(output1, output2)
  sim = 1-euclidean_distance.item()/1.5
  if(sim<.5):
    print(f'Not Accepted with similarity score of {sim}')
  else:
    print(f'Accepted with similarity score of {sim}')
else:
  print(f'Accepted with similarity score of {sim}')
