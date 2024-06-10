import torch
from torchvision import datasets, transforms
import json

# pixel color normalize stats per ImageNet dataset specification, same values for train and test.
pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]
img_crop_size = 224
img_side_px = 255
num_classes = 102

def produce_train_transforms(rotation=30, pixel_mean=pixel_mean, pixel_std=pixel_std, img_crop_size=img_crop_size):
    return transforms.Compose([transforms.RandomRotation(rotation),
                                       transforms.RandomResizedCrop(img_crop_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(pixel_mean, pixel_std)])

def produce_validation_transforms(img_side_px=img_side_px, img_crop_size=img_crop_size, pixel_mean=pixel_mean, pixel_std=pixel_std):
    return transforms.Compose([transforms.Resize(img_side_px),
                                      transforms.CenterCrop(img_crop_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(pixel_mean, pixel_std)])

def produce_test_transforms(img_side_px=img_side_px, img_crop_size=img_crop_size, pixel_mean=pixel_mean, pixel_std=pixel_std):
    return produce_validation_transforms(img_side_px=img_side_px, img_crop_size=img_crop_size, pixel_mean=pixel_mean, pixel_std=pixel_std)


def produce_data_loaders(data_dir='.', train_dir=None, valid_dir=None, test_dir=None, batch_size=64):
    train_dir = train_dir or data_dir + '/train'
    valid_dir = valid_dir or data_dir + '/valid'
    test_dir = test_dir or data_dir + '/test'

    train_transforms = produce_train_transforms()
    validation_transforms = produce_validation_transforms()
    test_transforms = produce_test_transforms()

    # configure the train and test data references to use the respective transforms
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # configure the train and test loaders, using "shuffle" on the training data to avoid bias resulting from processing order.
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return {
        'train_data': train_data,
        'validation_data': validation_data,
        'test_data': test_data,
        'trainloader': trainloader,
        'validationloader': validationloader,
        'testloader': testloader
    }


def get_category_to_name_map(filepath):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
