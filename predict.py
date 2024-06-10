
import argparse
from model_construction import load_model_bundle
from model_evaluation import predict

''' predict.py commandline program
    Basic Usage: 
    python predict.py /path/to/image checkpoint

    Return top K most likely classes:
    python predict.py input checkpoint --top_k 3

    Use a mapping of categories to real names:
    python predict.py input checkpoint --category_names cat_to_name.json
    * Note, we also save this in our checkpoint so this option serves to override what's in the checkpoint.

    GPU will be used automatically if available.
'''
def main():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('imagefile', type=str, default='./assets/sample-flower.jpg', help='Image to predict category on.')
    parser.add_argument('checkpoint', type=str, default='./checkpoint.pth', help='Checkpoint file to reconstruct model.')
    parser.add_argument('--top_k', type=int, default='5', help='Number of top predictions to report')
    parser.add_argument('--category_names', type=str, default=None, help='Override category names saved in checkpoint with an alternate JSON mapping')

    in_arg = parser.parse_args()

    imagefile = in_arg.imagefile
    checkpoint = in_arg.checkpoint
    top_k = in_arg.top_k
    category_names = in_arg.category_names
    
    bundle = load_model_bundle(checkpoint)
    model = bundle['model']
    cat_to_name = category_names or bundle['class_to_name']

    top_predictions = list(predict(image_path=imagefile, model=model, category_names=cat_to_name, topk=top_k))

    # print top_predictions
    print(f"Top Predictions (index, name, probability)")
    for category, name, probability in top_predictions:
        print(f"{category} {name} {probability*100:.3}%")


# Call to main function to run the program
if __name__ == "__main__":
    main()