# Convolutional Neural Network Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

