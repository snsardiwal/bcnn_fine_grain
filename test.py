import os
import sys
import numpy as np 
import cv2
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms as T
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
import argparse as ap
import sys
import os
sys.path.insert(0, "./models")
from train import BCNN





if __name__=='__main__':
	"""
	Input:
		model_path:Path to saved model
		test_path:Path to dir containing test images
		output_path:Path of the dir which will have the output file
	Output:
		output.txt
			image class class@subclass
			...
	"""

	parser = ap.ArgumentParser()
	parser.add_argument('--model_path',dest='model_path',required=True,help='Path to saved model')
	parser.add_argument('--test_path',dest='test_path',required=True,help='Path of directory containing test images')
	parser.add_argument('--output_path',dest='output_path',required=True,help='Path of the output file to be saved')

	args = parser.parse_args()
	model_path = args.model_path
	folder_path = args.test_path
	output_path = args.output_path

	transform = T.Compose([
	    T.Resize(448), 
	    T.CenterCrop(448), 
	    T.ToTensor(), 
	    T.Normalize(mean=(0.485, 0.456, 0.406),
	                std=(0.229, 0.224, 0.225)) 
	    ])

	idx_to_label = {
							0:'aircraft_1',1:'aircraft_2',2:'aircraft_3',3:'aircraft_4',
							4:'aircraft_5',5:'aircraft_6',6:'aircraft_7',
							7:'car_1',8:'car_2',9:'car_3',10:'car_4',11:'car_5',12:'car_6',
							13:'car_7',14:'car_8',
							15:'bird_1',16:'bird_2',17:'bird_3',18:'bird_4',19:'bird_5',20:'bird_6',
							21:'bird_7',22:'bird_8',23:'bird_9',24:'bird_10',25:'bird_11',
							26:'dog_1',27:'dog_2',28:'dog_3',29:'dog_4',30:'dog_5',
							31:'flower_1',32:'flower_2',33:'flower_3',34:'flower_4',35:'flower_5'
	}

	f = open(os.path.join(output_path,'output.txt'),'w')
	for root,dirs,imgs in os.walk(folder_path):
		print(root)
		for img in imgs:
			image= Image.open(os.path.join(root,img))
			
			model = BCNN()
			saved_param=torch.load(model_path).state_dict()
		
			param = torch.load(model_path)
			new_state_dict = OrderedDict()
			for k, v in param.state_dict().items():
				name = k[7:] # remove `module.`
				new_state_dict[name] = v
			
			model.load_state_dict(new_state_dict)
			
			input = transform(image)

			input=input.unsqueeze(0)
			output = model(input)
			index = torch.argmax(output,1)
			label = idx_to_label[index.item()]
			clss= label.split('_')
			print("%s %s@%s"%(img,clss[0],clss[1]))

			f.write("%s %s %s@%s\n" % (img,clss[0],clss[0],clss[1]))

		f.close()
		break
