import sys
import torch
import torchvision
import torchvision.models as models

import argparse
import os
import sys
import numpy as np 
from torch.utils import data
from torchvision.models.vgg import model_urls
import torch.nn as nn


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
import argparse 

from data_loader import Data


class BCNN(torch.nn.Module):
    """B-CNN.

    Network architechure of BCNN
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (num_classes).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.

    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: num_classes.
    """
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, 36)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.autograd.Variable of shape N*3*448*448.

        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        X = X.view(N, 512, 28**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 36)
        return X


class BCNNManager(object):
    """Manager class to train bilinear CNN.

    Attributes:
        _options: Hyperparameters.
        _path: Useful paths.
        _net: Bilinear CNN.
        _criterion: Cross-entropy loss.
        _solver: SGD with momentum.
        _scheduler: Reduce learning rate by a fator of 0.1 when plateau.
        _train_loader: Training data.
        _test_loader: Testing data.
        _train_path: Path to train annotation file
        _test_path: Path to test annotation file
    """
    def __init__(self, options):
        """Prepare the network, criterion, solver, and data.

        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        # Network.
        self._net = torch.nn.DataParallel(BCNN()).cuda()
        # Load the model from disk.
        #self._net.load_state_dict(torch.load(self._path['model']))
        print(self._net)
        # Criterion.
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
        self._solver = torch.optim.SGD(
            self._net.parameters(), lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._solver, mode='max', factor=0.1, patience=3, verbose=True,
            threshold=1e-4)

        self._train_path = os.path.join(self._options['text_path'],'train.txt')
        self._test_path = os.path.join(self._options['text_path'],'test.txt')

        #Dataloader
        transform = T.Compose([
        T.Resize(448), 
        T.CenterCrop(448), 
        T.ToTensor(), 
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)) 
        ])

        train_data = Data( train_path = self._train_path, aug_path = options['aug_data'], img_transform=transform)



        test_data = Data( train_path = self._test_path, aug_path = options['aug_data'], img_transform=transform)



        self._train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=self._options['batch_size'],drop_last=True, pin_memory=True,
                                                   shuffle=True,num_workers=4)

        self._test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=self._options['batch_size'],pin_memory=True,
                                                  shuffle=False,num_workers=4)

    def train(self):
        """Train the network."""
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')

        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for X, y in self._train_loader:
                # Data.
                X = torch.autograd.Variable(X.cuda())
                y = torch.autograd.Variable(y.cuda(async=True)).long()

                # Clear the existing gradients.

                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y)
                epoch_loss.append(loss.item())
                # Prediction.
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
               
                num_correct += torch.sum(prediction == y)
                # Backward pass.
                loss.backward()
                self._solver.step()
            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)

            
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                print('*')
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))

            torch.save(self._net, os.path.join(self._options['save_model'], "model_%s.pth" % t))
            

        print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))
    

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        self._net.train(False)
        num_correct = 0
        num_total = 0
        for X, y in data_loader:
            # Data.
            X = torch.autograd.Variable(X.cuda())
            y = torch.autograd.Variable(y.cuda(async=True)).long()

            # Prediction.
            score = self._net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # Set the model to training phase
        return 100 * num_correct / num_total




def main():
    """The main function."""
    
    parser = argparse.ArgumentParser(
        description='Train bilinear CNN.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int, required=True,
                        help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')

    parser.add_argument('--aug_data',dest='aug_data',required=True,help='Path to augmented data')
    parser.add_argument('--text_path',dest='text_path',required=True,help='Path to train-test text file')

    parser.add_argument('--save_model', dest='save_model', type=str, required=True,
                        help='Diretory in which to save the model')
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'aug_data' : args.aug_data,
        'save_model': args.save_model,
        'text_path': args.text_path
    }


    manager = BCNNManager(options)
    # manager.getStat()
    manager.train()


if __name__ == '__main__':

    main()