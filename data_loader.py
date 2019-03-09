import torch
from torch.utils import data
from torchvision.transforms import transforms
import numpy as np 
import os
import sys

from PIL import Image
import os
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms as T
from torch.utils.data.dataloader import default_collate
import datetime
import sys
import argparse as ap


class Data(data.Dataset):
	"""Geneartes image and label lists
	   input: Path to train text file, augmented data
	   return: image list, label list
	 """
	def __init__(self, train_path,aug_path,img_transform=None):
		
		self.fine_classes = {
						'aircraft_1':0,'aircraft_2':1,'aircraft_3':2,'aircraft_4':3,
						'aircraft_5':4,'aircraft_6':5,'aircraft_7':6,
						'car_1':7,'car_2':8,'car_3':9,'car_4':10,'car_5':11,'car_6':12,
						'car_7':13,'car_8':14,
						'bird_1':15,'bird_2':16,'bird_3':17,'bird_4':18,'bird_5':19,'bird_6':20,
						'bird_7':21,'bird_8':22,'bird_9':23,'bird_10':24,'bird_11':25,
						'dog_1':26,'dog_2':27,'dog_3':28,'dog_4':29,'dog_5':30,
						'flower_1':31,'flower_2':32,'flower_3':33,'flower_4':34,'flower_5':35
						}

		with open(train_path,'r') as f:
			lines = f.readlines()

			self.img_list = [os.path.join(aug_path,i.split()[0]) for i in lines]

			self.label_list = [float(self.fine_classes[i.split()[1]]) for i in lines]

		self.img_transform = img_transform

	def __len__(self):
		"""Denotes the total number of images"""
		return len(self.label_list)
		
	def __getitem__(self,index):
		imname = self.img_list[index]
		label = self.label_list[index]
		im = Image.open(imname).convert('RGB')

		if self.img_transform is not None:
			im  = self.img_transform(im)
		return im, label 

