import torch
import numpy as np
from PIL import Image
import time

# print out metrics bundle content
def print_metrics_record(metrics_record, include=None, separator='.. '):
    output = []
    if include is None or include.get('training'):
        output.extend([
            f"Epoch {metrics_record['epoch']}/{metrics_record['epochs']}",
            f"Steps: {metrics_record['steps']}",
            f"Train loss: {metrics_record['running_loss']:.3f}",
            f"Train step duration: {metrics_record['step_duration']:.3f}"
        ])
    
    if include is None or include.get('accuracy'):
        output.extend([
            f"Loss: {metrics_record['loss']:.3f}",
            f"Accuracy: {metrics_record['accuracy']*100:.3f}%",
            f"Accuracy Duration: {metrics_record['accuracy_duration']:.3f}"
        ])
    
    print(separator.join(output))

# applies the model to the dataloader series - used by both validation and testing
# qualifier can be any string, but "validation" and "testing" are supported by the metrics printout.
def check_accuracy(model, criterion, dataloader, metrics_record={}, device_name=None, ):
    loss = 0
    accuracy = 0

    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)

    # turn off training during eval to skip training only strategies like Dropout.
    model.eval()
    
    # disable gradient tracking during evaluation to improve performance when we know we won't call backward()
    with torch.no_grad():
        start_time = time.time()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # run the model on a batch
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            # capture some statistics on how the model performed on the data
            loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    metrics_record[f"accuracy_duration"] = time.time() - start_time
    # get ready for the next training iteration
    model.train() 

    metrics_record[f"loss"] = loss/len(dataloader)
    metrics_record[f"accuracy"] = accuracy/len(dataloader)


def process_image(image, short_size=256, img_crop_size=224):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    shorter_side = image.height if image.height < image.width else image.width
    scaling = shorter_side // short_size
    new_height = image.height * scaling
    new_width = image.width * scaling

    im_resized = image.resize((new_width, new_height))
    edge_dist = img_crop_size // 2
    center_x = new_width // 2
    center_y = new_height // 2
    upper = center_y - edge_dist
    left = center_x - edge_dist 
    lower = center_y + edge_dist
    right = center_x + edge_dist
    im_cropped = im_resized.crop([left, upper, right, lower])

    # comment out dimension logging
    # print(f"cropped image dims: h({im_cropped.height}) w({im_cropped.width})")

    im = np.array(im_cropped) / 255.0
    im = (im - pixel_mean) / pixel_std
    im = im.transpose((2, 0, 1))

    # comment out dimnsion logging
    # print(f"numpy image array shape: {im.shape}")

    return im


def predict(image_path, model, category_names, device_name=None, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        Return a ziped list of the topk picks: (category_index, category_name, probability)
    '''
    processed_image = None
    with Image.open(image_path) as pil_image:
        processed_image = process_image(pil_image)

    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    torch_image = torch.from_numpy(processed_image).float().unsqueeze(0)
    torch_image = torch_image.to(device)
    model.eval()
    model.to(device)
    logps = model.forward(torch_image)
    ps = torch.exp(logps)
    model.train()

    top_values, top_indices = torch.topk(ps, k=topk)
    top_names = [category_names[str(idx)] for idx in top_indices[0].tolist()]

    return zip(top_indices[0].tolist(), top_names, top_values[0].tolist())   