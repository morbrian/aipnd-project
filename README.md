# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

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


## Extracting cat and dog data:

```
unzip Cat_Dog_data.zip -x '__MACOSX/*'
rm Cat_Dog_data/.DS_Store
# maybe could have used a second "-x" ?
```
