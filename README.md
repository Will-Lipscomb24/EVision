# EVision: Image Reconstruction Using Event Images


You will need to install necessary dependencies as well as the open-source repository provided by Prophesee that provides necessary classes.
We will walk through the following process:
* Package and OpenEB installation
* Dataset Generation and Organization
* Training and Inference


## Package and OpenEB Installation
The full instructions for installing OpenEB can be found at https://github.com/prophesee-ai/openeb , but I will walk through a quick install:


* Clone the OpenEB repository
```bash
git clone https://github.com/prophesee-ai/openeb.git --branch 5.2.0
```
* Install OS dependencies:
```bash
sudo apt update
sudo apt -y install apt-utils build-essential software-properties-common wget unzip curl git cmake
sudo apt -y install libopencv-dev libboost-all-dev libusb-1.0-0-dev libprotobuf-dev protobuf-compiler
sudo apt -y install libhdf5-dev hdf5-tools libglew-dev libglfw3-dev libcanberra-gtk-module ffmpeg
```

* We will create a Conda environment to manage the dependencies:
```bash
conda create --name evision python=3.11
conda activate evision
```

```bash 
pip install -r OPENEB_SRC_DIR/utils/python/requirements_openeb.txt 
pip install -r OPENEB_SRC_DIR/utils/python/requirements_pytorch_cuda.txt
```

* Now we will compile. 
	1. In your openeb dir run: `mkdir build & cd build`
	2. Generate cmake `cmake .. -DPython3_EXECUTABLE=$(which python) -DCOMPILE_PYTHON3=ON`
	3. Compile: `make -j$(nproc)`

* We will use OpenEB directly from the build folder. This will allow for the source code to be modified if necessary. To do so we need to add a source line to `~/.bashrc`.
```bash
source <path to build folder>/utils/scripts/setup_env.sh
```

* Now the environment is setup so that OpenEB modules can be imported into scripts.
# Dataset Generation and Organization
The dataset used for the training & validation pipelines is the MSCOCO 2017 train images(118K/18GB) and 2017 val images(5K/1GB).
Only a subset of the full training dataset is used (~10K).
* Download the dataset that you wish to train on

### Data Modifications:
There are 3 ways in which we will modify the original data:
1. Reshaping all of the training imagery to be the same size.
2. Changing the exposure of the ground truth images to generate the models input image.
3. Using the ground truth imagery to generate small event-camera clips using the installed Metavision OpenEB tools.

## Dataset Generation
The scripts for reshaping and changing the exposure are located in
EVision/src/utils. Once this is complete, run the script **run_event_simulation.py** located in EVision/scripts. 
* Note: You will need to change the paths to reflect your data locations and you will need to specify the number of image files you wish to simulate.

## Training and Inference
