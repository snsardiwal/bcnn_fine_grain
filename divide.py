from random import shuffle
import os
import argparse as ap


if __name__ == "__main__":
	"""
	Generates train and test text files
	Input: Path to augmented data, save path and train-test ratio.
	Output:Saves train and test text files in save path
		train.txt
			imagename.jpg class_subclass
			...
		test.txt
			imagename.jpg class_subclass
			...
	"""

	parser = ap.ArgumentParser()
	parser.add_argument('--aug_data',dest= 'aug_data', required = True, help='Path to augmented data')
	parser.add_argument('--save',dest= 'save', required = True, help='Path for saving train and test files')
	parser.add_argument('--ratio',dest= 'ratio', required = True,type=float, help='fraction of train set')
	args = parser.parse_args()

	folderPath = args.aug_data
	save_path = args.save
	ratio = args.ratio

	assert ratio<1.0

	for root,dirs,imgs in os.walk(folderPath):

		train_index = int(ratio*len(imgs))

		train_filenames = imgs[:train_index]
		test_filenames = imgs[train_index:]
		break

	with open(os.path.join(save_path,'train.txt'), 'w') as f:
		for item in train_filenames:
			classname = item.split('_')[0]
			subclass = item.split('_')[1]
			f.write("%s %s_%s\n" % (item,classname,subclass))
	with open(os.path.join(save_path,'test.txt'), 'w') as f:
		for item in test_filenames:
			classname = item.split('_')[0]
			subclass = item.split('_')[1]
			f.write("%s %s_%s\n" % (item,classname,subclass))