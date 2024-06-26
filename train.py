from model_construction import construct_model_from_parameters, produce_optimizer, produce_criterion, save_model
from feature_loading import produce_data_loaders, get_category_to_name_map
from model_training import train_model
import argparse
import sys

''' train.py commandline program
    Basic Usage: 
    python train.py data_dir --save_dir save_directory

    Choose Architecture:
    python train.py data_dir --arch "vgg16"
    python train.py data_dir --arch "alexnet"

    Set Hyperparameters
    python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 5

    GPU will be used automatically if available, or --gpu / --cpu options will choose the sepcific processing type.
'''
def main():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default='./flowers', 
                        help='data folder organized by train, valid, test subfolders')
    parser.add_argument('--cat_to_name', type=str, default='./cat_to_name.json', 
                        help='path to json file with category to name mappings')
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='path to folder where our checkpoint files is saved')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg13', 'vgg16', 'vgg19', 'alexnet', 'densenet121'], 
                        help='CNN model architecture to use')
    parser.add_argument('--classifier', type=str, default='custom', choices=['vgg_inspired_short', 'vgg_inspired_long', 'custom'], 
                        help='Classifier architecture to apply to outputs of selected pretrained architecture')
    parser.add_argument('--num_classes', type=int, default='102', 
                        help='Set the number of classes expected for the training dataset')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], 
                        help='Select the preferred optimizer algorithm for the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='learning rate for the optimizer')
    parser.add_argument('--criterion', type=str, default='NLLLoss', choices=['NLLLoss', 'CrossEntropyLoss'], 
                        help='Set the preferred criterion algorithm')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='number of epochs to use during training')
    parser.add_argument('--print_every', type=int, default=5, 
                        help='how often to print out accuracy metrics during training')
    parser.add_argument('--gpu', action='store_true',
                    help='use GPU for training and validation, only supports GPUs with CUDA, cannot be specified with --cpu')
    parser.add_argument('--cpu', action='store_true', 
                    help='use CPU for training and validation, cannot be specified with --gpu')
    parser.add_argument('--print_arch', action='store_true', 
                    help='Display the model architecture before training.')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[512], 
                        help='List of hidden units for each hidden layer')
    

    in_arg = parser.parse_args()

    data_dir = in_arg.data_dir
    cat_to_name_file = in_arg.cat_to_name
    save_dir = in_arg.save_dir
    model_id = in_arg.arch
    classifier_id = in_arg.classifier
    num_classes = in_arg.num_classes
    optimizer_id = in_arg.optimizer
    learnrate = in_arg.learning_rate
    criterion_id = in_arg.criterion
    hidden_units = in_arg.hidden_units
    epochs = in_arg.epochs
    cpu = in_arg.cpu
    gpu = in_arg.gpu
    print_arch = in_arg.print_arch

    if cpu and gpu:
        print('Cannot specify both --cpu and --gpu')
        sys.exit(1)
    
    device_name = 'cpu' if cpu else 'cuda' if gpu else None

    # create data loaders to access the features and category metadata
    dataloaders = produce_data_loaders(data_dir)
    category_to_name_map = get_category_to_name_map(cat_to_name_file)

    # construct the model, optimizer, criterion specified by commandline parameters
    model = construct_model_from_parameters(model_id, classifier_id, num_classes, hidden_units)
    optimizer = produce_optimizer(optimizer_id, model.classifier, learnrate=learnrate)
    criterion = produce_criterion(criterion_id)

    if print_arch:
        print('__Model')
        print(model)
        print('__Optimizer')
        print(optimizer)
        print('__Criterion')
        print(criterion)

    # train the model
    train_model( 
        model, 
        trainloader=dataloaders['trainloader'],
        validationloader=dataloaders['validationloader'],
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device_name=device_name
    )

    # if save_dir option was specified
    # save the model state, along with metadata for evaluation objects and feature metadata
    if save_dir:
        checkpoint_file = f"{save_dir}/checkpoint.pth"
        save_model(
            model=model,
            filename=f"{save_dir}/checkpoint.pth",
            class_to_name=category_to_name_map,
            train_data=dataloaders['train_data'],
            optimizer=optimizer,
            criterion=criterion
        )


# Call to main function to run the program
if __name__ == "__main__":
    main()

