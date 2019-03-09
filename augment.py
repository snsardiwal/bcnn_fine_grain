import numpy as np 
import cv2
import os
import argparse as ap



if __name__ == '__main__':
	"""
	Performs data augmentation
	Input: Path to data, path to save
		Data directory format
			-root
				--class1
					--subclass1
					--subclass2
					...
				--class2
					--subclass1
					--subclass2
					...
				...
	Output: Saves images in save directory
			Saved image format:
				class_subclass_int_int.jpg
	"""
	#Parse the arguments
	parser = ap.ArgumentParser(description='Data Augmentation')
	parser.add_argument('--data',dest='data',required=True,help='Path to dataset')
	parser.add_argument('--save',dest='save',required=True,help='Path to save')

	args = parser.parse_args()

	folderPath = args.data
	augment_path = args.save

	#Traverse the directory
	for root,dirs,imgs in os.walk(folderPath):
		
		for classname in dirs:
			
			for root1,dirs1,imgs1 in os.walk(os.path.join(folderPath,classname)):
				for subclass in dirs1:
					count=1
				
					for root2,dirs2,imgs2 in os.walk(os.path.join(folderPath,classname,subclass)):
						
						for image in imgs2:
							img = cv2.imread(os.path.join(folderPath,classname,subclass,image))
							num_cols,num_rows = img.shape[:2]


							#Scaled image
							scale_img = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

							#Rotated image
							rotation_matrix = cv2.getRotationMatrix2D((num_cols/2,num_rows/2),40,1)
							rot_img = cv2.warpAffine(img,rotation_matrix,(num_cols,num_rows))

							#Horizontal flip
							horizontal_img = cv2.flip( img, 0 )
							#Vertical flip
							vertical_img = cv2.flip( img, 1 )
							

							#Noisy image
							gauss = np.random.randn(num_cols,num_rows,3)
							gauss = gauss.reshape(num_cols,num_rows,3)        
							noisy_img = img + img * gauss

							#Smoothed image(gaussian blurring)
							vals = len(np.unique(img))
							vals = 3**np.ceil(np.log2(vals))
							kernel = np.ones((5,5),np.float32)/25
							smooth_img = cv2.filter2D(img,-1,kernel)

							#Save images
							save_name = classname + '_' + subclass + '_' + str(count)
							cv2.imwrite(augment_path + '/' + save_name + '_1.jpg', scale_img)
							cv2.imwrite(augment_path + '/' + save_name + '_2.jpg', horizontal_img)
							cv2.imwrite(augment_path + '/' + save_name + '_3.jpg', vertical_img)
							cv2.imwrite(augment_path + '/' + save_name + '_4.jpg', rot_img)
							cv2.imwrite(augment_path + '/' + save_name + '_5.jpg', noisy_img)
							cv2.imwrite(augment_path + '/' + save_name + '_6.jpg', smooth_img)
							
							count=count+1

						break
				break
		break
		