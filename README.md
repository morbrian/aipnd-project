# AI Programming with Python Project

## Part-1: Jupyter Notebook

The notebook is the file `Image Classifier Project.ipynb`

The HTML export of the notbook after running all cells is: 
* [Image_Classifier_Project.html](Image_Classifier_Project.html)

Note that the checkpoint format saved by the notebook is not compatible with the commandline application.
* Notebook saves file as `checkpoint-nb.pth` to avoid overwriting the CLI saved checkpoints.
* The CLI app format evolved beyond the Jupyter code snippets and the two formats diverged.


## Part-2: Commandline Application

Notes about code organization.

1. train.py - commandline application for training a network.
1. predict.py - commandline application for predicting a category for an input image.
1. feature_loading.py - utilities for loading data features.
1. model_construction.py - utilities for saving and restoring a model checkpoint.
1. model_training.py - primary controller for how the network gets trained.
1. model_evaluation.py - functions for evaluating model performance and predicting image categories.

## Staging Data

This projected used the provided Flowers data set, and sample commandlines assume this is extracted to a `./flowers` subfolder.
* Downloadable here: [Download Flowers Dataset](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)

## Train

* Basic Usage: `python train.py ./flowers`

* Set directory to save checkpoints: 
    ```
    mkdir custom
    python train.py data_dir --save_dir ./custom
    ```

* Choose architecture: `python train.py ./flowers --arch "vgg13"`

* Set hyperparameters: `python train.py ./flowers --learning_rate 0.01 --hidden_units 512 --epochs 20`

* Use GPU for training: `python train.py data_dir --gpu`
    * Note that `gpu` is used by default when available even without this option.
    * Can also specify `--cpu` to downgrade to using only cpu.

### Extended Training Options

* Print architecture before Training: `python train.py ./flowers --print_arch`

* Select altnerate optimizer and loss functions: `python train.py ./flowers --criterion CrossEntropyLoss --optimizer SGD`

* Specify prepared classifier by identifier: python train.py ./flowers --classifier vgg_inspired_short

* Specify category names file, it will be saved with checkpoint: `python train.py ./flowers --save_dir . --cat_to_name cat_to_name.json`

## Predict

* Basic usage: `python predict.py ./flowers/test/26/image_06526.jpg checkpoint.pth`

* Return top K most likely classes: `python predict.py ./flowers/test/26/image_06526.jpg checkpoint.pth --top_k 3`

* Use a mapping of categories to real names: `python predict.py ./flowers/test/26/image_06526.jpg checkpoint.pth --category_names cat_to_name.json`
    * Note that the default mapping was also saved to the checkpoint file during training.
    * Specifying --category_names on prediction will override the mapping stored in the checkpoint.

* Use GPU for inference: `python predict.py ./flowers/test/26/image_06526.jpg checkpoint.pth --gpu`
    * Note that `gpu` is used by default when available even without this option.
    * Can also specify `--cpu` to downgrade to using only cpu.

### Extended Prediction Options

* Print architecture loaded from checkpoint before prediction: `python predict.py ./flowers/test/26/image_06526.jpg checkpoint.pth --print_arch`

# Appendices

Additional notes about getting an environment setup.

## Local Configuration

There are many possible configurations of Python, Jupyter, PyTorch, NVidia with great documentation on the related websites and across the internet.

This notebook section captures how we configured a local system for working on this project.

**Basesline System:** 
* OS: WSL-Ubunu 22.04 under Windows 11
* Conda 22.9.0
* Python 3.12.3
* NVidia SMI 555.52.01 / CUDA Version 12.5

All of these steps are performed from the WSL Ubuntu commandline.

1. Create your conda environment
    * We assume you've already installed conda, not covered in these steps.
    * Install Reference: https://docs.anaconda.com/free/anaconda/install/linux/
    
    ```bash
    # if "python" option excluded, will grab latest python, which is currently also 3.12
    conda create -n udacity-aipnd python=3.12

    # be sure to activate!
    conda activate udacity-aipnd
    ```


2. Get the basic python packages
    ```bash
    pip install notebook numpy pandas matplotlib pillow scikit-learn argparse
    # optional: if you do not plan to use a GPU, you can just install torch and torchvision like this and stop here.
    # otherwise, skip this and proceed to step (2)
    pip install torch torchvision
    ```


3. GPU Drivers and CUDA Library

    Review the status of your NVidia configuration with the `nvidia-smi` command.
    * If the `nvidia-smi` command is not availabel, the next steps will document installing it.
    * Our goal is to be compatible with PyTorch, which documents configurations here: https://pytorch.org
    * The `12.5` version we installed is later than the listed compatible versions, but we'll stick with it until we have a reason not to.
    
    *Tips:* You might find references online to `nvcc --version` as a way to dicover your CUDA Version.
        * Read more about why this is misleading: https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi
        * Summary: trust the output of `nvidia-smi` as the ground truth for what your system is configured to support.

    NVidia on WSL-Ubuntu is very common and well documented.
    * https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network
    * That longish link filters the options to exactly what we did on our WSL Ubuntu.

    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-5
    ```

    After install expect the `nvidia-smi` command to be available and to print a table of configuration information listing `CUDA 12.5`
    * I restarted my WSL Ubuntu after the install, I'm not sure if it was a necessary step.


4. Install PyTorch
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```


5. Verify PyTorch Configuration
    Expect to see `True` in the output of this at the commandline.
    ```bash
    python -c 'import torch; print(f"GPU Available: {torch.cuda.is_available()}")'
    ```

6. System Ready!


## Extracting data without macos metadata:

```
unzip Cat_Dog_data.zip -x '__MACOSX/*'
rm Cat_Dog_data/.DS_Store
# maybe could have used a second "-x" ?
```
