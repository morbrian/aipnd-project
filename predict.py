
import argparse
from model_construction import load_model_bundle
from model_evaluation import predict, print_predictions
from feature_loading import get_category_to_name_map
import sys

''' predict.py commandline program
    Basic Usage: 
    python predict.py /path/to/image checkpoint

    Return top K most likely classes:
    python predict.py input checkpoint --top_k 3

    Use a mapping of categories to real names:
    python predict.py input checkpoint --category_names cat_to_name.json
    * Note, we also save this in our checkpoint so this option serves to override what's in the checkpoint.

    
'''
def main():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('imagefile', type=str, default='./assets/sample-flower.jpg', 
                        help='Image to predict category on.')
    parser.add_argument('checkpoint', type=str, default='./checkpoint.pth', 
                        help='Checkpoint file to reconstruct model.')
    parser.add_argument('--top_k', type=int, default=1, 
                        help='Number of top predictions to report')
    parser.add_argument('--category_names', type=str, default=None, 
                        help='Override category names saved in checkpoint with an alternate JSON mapping')
    parser.add_argument('--gpu', action='store_true',
                        help='use GPU for training and validation, only supports GPUs with CUDA, cannot be specified with --cpu')
    parser.add_argument('--cpu', action='store_true', 
                        help='use CPU for training and validation, cannot be specified with --gpu')

    in_arg = parser.parse_args()

    imagefile = in_arg.imagefile
    checkpoint = in_arg.checkpoint
    top_k = in_arg.top_k
    category_names = get_category_to_name_map(in_arg.category_names)
    cpu = in_arg.cpu
    gpu = in_arg.gpu
    
    if cpu and gpu:
        print('Cannot specify both --cpu and --gpu')
        sys.exit(1)
    
    device_name = 'cpu' if cpu else 'cuda' if gpu else None
    bundle = load_model_bundle(checkpoint)
    model = bundle['model']
    cat_to_name = category_names or bundle['class_to_name']

    top_predictions = list(predict(image_path=imagefile, model=model, category_names=cat_to_name, topk=top_k, device_name=device_name))

    # print identified image category
    category, name, probability = top_predictions[0]
    print(f"Prediction: {category} {name} {probability*100:.3}%\n")

    # print top_predictions
    if top_k > 1:
        print_predictions(top_predictions, title="Top Predictions")


# Call to main function to run the program
if __name__ == "__main__":
    main()