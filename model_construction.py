import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict
import json

vgg_class_input = 25088
alexnet_class_input = 9216
densenet_class_input = 1024

def produce_pretrained_model(model_id, freeze_base_model=True):
    ''' Produce a pretrained model for the given model_id

        Arguments
        ---------
        model_id: string identifier for a pretrained model we know how to reconstruct.
        freeze_base_model: when true we set requireds_grad=False for the pretrained model layers.
    '''
    model_instance = None
    if model_id == 'vgg16':
        model_instance = models.vgg16(weights=models.vgg.VGG16_Weights.DEFAULT)
        model_instance.input_count = vgg_class_input
    if model_id == 'vgg19':
        model_instance = models.vgg19(weights=models.vgg.VGG19_Weights.DEFAULT)
        model_instance.input_count = vgg_class_input
    if model_id == 'alexnet':
        model_instance = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model_instance.input_count = alexnet_class_input
    if model_id == 'densenet121':
        model_instance = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model_instance.input_count = densenet_class_input
    
    if model_instance:
        if freeze_base_model:
            # Freeze parameters so we don't backprop through them
            for param in model_instance.parameters():
                param.requires_grad = False
        model_instance.model_id = model_id
        return model_instance

    raise Exception(f"Usupported model identifier: {model_id}")


def produce_classifier(classifier_id, output_count=102, input_count=512):
    ''' Produce identified instance of Sequential defining the layer architecture for the classifier
        
        Arguments
        ---------
        classifier_id: string identifier of the classifier
        output_count: defines expected size of the output layer
    '''
    classifier_instance = None
    if classifier_id == 'vgg_inspired_short':
        classifier_instance = nn.Sequential(OrderedDict([
            ('fc1-from-vgg16', nn.Linear(in_features=input_count, out_features=1024, bias=True)),
            ('act1-from-vgg16', nn.ReLU(inplace=True)),
            ('reg1-from-vgg-16', nn.Dropout(p=0.5, inplace=False)),
            ('fc2', nn.Linear(1024, output_count)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    if classifier_id == 'vgg_inspired_long':
        classifier_instance = nn.Sequential(OrderedDict([
            ('fc1-from-vgg16', nn.Linear(in_features=input_count, out_features=1024, bias=True)),
            ('act1-from-vgg16', nn.ReLU(inplace=True)),
            ('reg1-from-vgg-16', nn.Dropout(p=0.5, inplace=False)),
            ('fc2-from-vgg16', nn.Linear(in_features=4096, out_features=4096, bias=True)),
            ('act2-from-vgg16', nn.ReLU(inplace=True)),
            ('reg2from-vgg16', nn.Dropout(p=0.5, inplace=False)),
            ('fc3-from-vgg16', nn.Linear(in_features=4096, out_features=1024, bias=True)),
            ('fc4', nn.Linear(4096, 1024)),
            ('act4', nn.ReLU()),
            ('reg4', nn.Dropout(0.5)),
            ('fc5', nn.Linear(1024, output_count)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    
    if classifier_instance:
        classifier_instance.classifier_id = classifier_id
        classifier_instance.output_count = output_count
        classifier_instance.input_count = input_count
        return classifier_instance

    raise Exception(f"No classifier matches id {classifier_id}")


def produce_optimizer(optimizer_id, classifier, learnrate=0.001):
    ''' Produce the optimizer specified by optimizer_id
        Throws exception if an unsupported optimizer is specified.
    '''
    optimizer_instance = None
    if optimizer_id == 'Adam':
        optimizer_instance = optim.Adam(classifier.parameters(), lr=learnrate)
    if optimizer_id == 'SGD':
        optimizer_instance = optim.SGD(classifier.parameters(), lr=learnrate)
    
    if optimizer_instance:
        optimizer_instance.optimizer_id = optimizer_id
        optimizer_instance.learnrate = learnrate
        return optimizer_instance
    
    raise Exception(f"Unsupported optimizer requested {optimizer_id}")


def produce_criterion(criterion_id):
    ''' Produce the criterion instance for the specified id.
    '''
    criterion_instance = None
    if criterion_id == 'NLLLoss':
        criterion_instance = nn.NLLLoss()
    if criterion_id == 'CrossEntropyLoss':
        criterion_instance = nn.CrossEntropyLoss()
    
    if criterion_instance:
        criterion_instance.criterion_id = criterion_id 
        return criterion_instance 
    
    raise Exception(f"Unsupported criterion request {criterion_id}")


def construct_model_from_parameters(model_id, classifier_id=None, num_classes=102):
    model = produce_pretrained_model(model_id)
    if classifier_id:
        model.classifier = produce_classifier(classifier_id, output_count=num_classes, input_count=model.input_count)
    
    return model

def save_model(model, filename, class_to_name=None, train_data=None, optimizer=None, criterion=None):
    ''' Create a checkpoint file in a format we can restore later
        Checkpoint structure:
            model_id: identifier of a pretrained basemodel we can reconstruct
            classiifier_id: identifier of a classifier archticture we can reconstruct
            output_count: number of output categories the classifier should reproduce
            state_dict: standard format for torch serialized model.state_dict with model parameters
            integrity_check: text representation of model used to verify the reconstructed model architecture matches the saved model
            data_mapping_metadata: parameters used for data loading and category naming
            training_metadata: supporting properties to reconsruct necessary training objects like the optimizer.
            evaluation_metadata: supporting properties to reconstruct necessary objects for validation and testing, like criterion.
        
        Arguments
        ---------
        model: the trained model
        filename: name of file to save model state to
    '''
    checkpoint = {
        'model_id': model.model_id,
        'classifier_id': model.classifier.classifier_id,
        'output_count': model.classifier.output_count,
        'input_count': model.classifier.input_count,
        'state_dict': model.state_dict(),
        'integrity_check': str(model),
        'data_mapping_metadata': {
            'class_to_idx': train_data.class_to_idx if train_data else None,
            'class_to_name': json.dumps(class_to_name) if class_to_name else None
        },
        'training_metadata': {
            'optimizer_id': optimizer.optimizer_id if optimizer else None,
            'learnrate': optimizer.learnrate if optimizer else None,
            'optimizer_state': optimizer.state_dict() if optimizer else None,
        },
        'evaluation_metadata': {
            'criterion_id': criterion.criterion_id if criterion else None
        }
    }
    torch.save(checkpoint, filename)


def load_model_bundle(filepath):
    ''' Load model from a checkpoint 
        Checkpoint structure:
            model_id: identifier of a pretrained basemodel we can reconstruct
            classiifier_id: identifier of a classifier archticture we can reconstruct
            output_count: number of output categories the classifier should reproduce
            input_count: number of inputs the classifier can expect from pretrained architecture
            state_dict: standard format for torch serialized model.state_dict with model parameters
            integrity_check: text representation of model used to verify the reconstructed model architecture matches the saved model
            data_mapping_metadata: parameters used for data loading and category naming
            training_meta_data: supporting properties to reconsruct necessary training objects like the optimizer.
            evaluation_meta_data: supporting properties to reconstruct necessary objects for validation and testing, like criterion.

        Arguments
        ---------
        filepath: filename of checkpoint file to load.

        Throws
        ------
        Exception if restored model does not match integrity_check field of checkpoint file.
        Exception if we don't support reconsructing the model_id specified in checkpoint file.
        Exception if we don't support reconsructing the classifier_id specified in checkpoint file.
    '''
    checkpoint = torch.load(filepath)
    # we load the pretrained model identified by model_id
    model_id = checkpoint['model_id']
    base_model = produce_pretrained_model(model_id)
    # we set classifier to our layer Sequence referenced by classifier
    classifier_id = checkpoint['classifier_id']
    output_count = checkpoint['output_count']
    input_count = checkpoint['input_count']
    base_model.classifier = produce_classifier(classifier_id, output_count=output_count, input_count=input_count)
    base_model.load_state_dict(checkpoint['state_dict'])
    # verify the archtitecture we loaded matches what we saved
    integrity_check = checkpoint['integrity_check']
    if integrity_check != str(base_model):
        raise Exception(f"Loaded model architecture differes from what was saved.")
    
    criterion = None
    optimizer = None
    class_to_name = None 
    class_to_idx = None

    data_mapping_metadata = checkpoint.get('data_mapping_metadata', None)
    if data_mapping_metadata:
        class_to_name_data = data_mapping_metadata.get('class_to_name', None)
        class_to_name = json.loads(class_to_name_data) if class_to_name_data else None 
        class_to_idx = data_mapping_metadata.get('class_to_idx', None)
    
    training_metadata = checkpoint.get('training_metadata', None)
    if training_metadata:
        optimizer_id = training_metadata.get('optimizer_id', None)
        learnrate = training_metadata['learnrate'] if optimizer_id else None
        optimizer = produce_optimizer(training_metadata['optimizer_id'], base_model.classifier, learnrate=learnrate) if optimizer_id else None
        if optimizer: 
            optimizer.load_state_dict(training_metadata['optimizer_state'])
    
    evaluation_metadata = checkpoint.get('evaluation_metadata', None)
    if evaluation_metadata:
        criterion_id = evaluation_metadata.get('criterion_id', None)
        criterion = produce_criterion(criterion_id) if criterion_id else None

    return {
        'model': base_model,
        'criterion': criterion,
        'optimizer': optimizer,
        'class_to_name': class_to_name,
        'class_to_idx': class_to_idx,
    }

