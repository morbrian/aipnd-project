import time
import torch
from model_evaluation import check_accuracy, print_metrics_record

def train_model(model, trainloader, validationloader, optimizer, criterion, device_name=None, epochs=1, print_every=5):

    # we use the metrics array to store a history of how the accuracy changed during a completed training cycle
    # this is useful for post-training analisys
    # a single metrics record will be a dictionary with keys like: { epoch, step, running_loss, validation_loss, accuracy }
    # we start with a new metrics record added to the aray every "print_every"
    metrics = []
    def produce_metrics_record():
        return { 
            'running_loss': 0, 
            'epochs': epochs, 
            'step_batch_start': time.time(), 
            'step_batch_size': print_every,
            'device': device_name
        }

    metrics_record = produce_metrics_record()
    steps = 0
    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)
    print(f"Traing device: {device_name}")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps +=1
            inputs, labels = inputs.to(device), labels.to(device)

            # capture logarithm of probabilities in the forward step
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            # clear out the gradients before our back propagation to help avoid introducing unintended bias from or previous iterations
            optimizer.zero_grad()
            loss.backward()

            # update parameters with lessons learned in the back prop calculations
            optimizer.step()

            metrics_record['running_loss'] += loss.item()

            if steps % print_every == 0:
                metrics_record['step_duration'] = (time.time() - metrics_record['step_batch_start']) / metrics_record['step_batch_size']
                metrics_record['train_loss'] = metrics_record['running_loss']/metrics_record['step_batch_size']
                metrics_record['epoch'] = epoch+1
                metrics_record['steps'] = steps

                # check accuracy and record metrics
                check_accuracy(model, criterion, validationloader, metrics_record=metrics_record, device_name=device_name)

                # save record to history
                metrics.append(metrics_record)
                
                # print metrics summary for recent batch of steps
                print_metrics_record(metrics_record)

                # start a new record and maintain the running_loss data
                metrics_record = produce_metrics_record()
                metrics_record['running_loss'] = metrics_record['running_loss']