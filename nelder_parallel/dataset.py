import torch
import random
from torchvision import datasets
from torchvision import transforms
import numpy as np

def get_data(batch_size):

    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                          std=[0.2675, 0.2565, 0.2761])
    transform_train = transforms.Compose([
                                transforms.RandomCrop(32, padding = 4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                normalize])
    transform_test = transforms.Compose([
                                transforms.ToTensor(), 
                                normalize])

    train_dataset = datasets.CIFAR100(
                        root = "cifar",
                        train = True, 
                        download = True,
                        transform = transform_train
                        )
    test_dataset = datasets.CIFAR100(
                        root = "cifar",
                        train = False, 
                        download = False,
                        transform = transform_test
                        )

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)
    
    return train_data, test_data