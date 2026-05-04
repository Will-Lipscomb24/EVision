# EVision: Image Reconstruction Using Events and Images
This repo serves as a computer vision pipeline where you can generate synthetic event data, train, and inference to reconstruct poorly exposed gray-scale images using the event data.

## Contents
* [Installation](#installation)
* [Data Generation](#data-generation)
* [Training](#training)
* [Testing & Analysis](#testing--analysis)


## Installation
[OpenEB](https://github.com/Will-Lipscomb24/openeb/tree/new_sim) is an open-source toolkit that can be used to generate synthetic event data, interface with real event cameras, and perform computer vision tasks using event streams. The link provided is a forked version that allows the user to specify additional parameters. 
To clone EVision and OpenEB together, run:
```
git clone --recursive https://github.com/Will-Lipscomb24/EVision
```
Visit the OpenEB link to install the additional required dependencies and compile the source code. After you clone you will need to navigate to the `openeb` directory before compiling.
It is recommended to use a dependency manager such as conda or venv to manage your packages, but it is not strictly necessary.

### Dependencies for Using IDS Event Camera
If you want to use an IDS event camera, the plugin will need to be installed and added to the path. There exists different camera plugins depending on the manufacturer of event camera that you are using. Focusing specifically on the IDS event camera, you will need to navigate to the [software](https://www.ids-imaging.us/download-details/1011378.html?os=linux&version=&bus=64&floatcalc=#) webpage and download the correct plugin depending on the version of Ubuntu that you are using. Follow the install instructions provided in the download and be sure to setup the udev rules as instructed. As instructed in the **Compilation** section of the openEB repository, in order for openEB to find the plugin, export the installed plugin path in your `~/.bashrc` directory:
```
export MV_HAL_PLUGIN_PATH=<plugin_path>
```
The plugin path after installing the IDS plugin should resemble `/opt/ueye-evs/lib/ids/ueye_evs/hal/plugins`. Note that the setup of the IDS event camera plugin isn't necessary for the generation on synthetic events.
To test if the installations succeeded, plugin the camera whos plugin was linked and run:
```
metavision_viewer
```
This should show a live-stream of the event camera output.

## Data Generation
The dataset that was used to generate synthetic events was a 15k subset of the 2017 open-source **MSCOCO** dataset. To download it run:
```
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```
There are a few steps that need to take place to get the data in the proper form for training. This includes renaming the files, scaling to a unique size, converting to gray-scale, and over-exposing the input images.

All of these functions exist in `EVision/src/utils`. The config file has parameters under `model` and `training` that need to be set by the user. 

| Parameter | Description |
| :--- | :--- |
| `model: desired_images` | Subset of images from the downloaded MSCOCO dataset that will be trained on |
| `model: height` | Image height used for training and reconstruction |
| `model: width` | Image width used for training and reconstruction |
| `training: type` | Set as training or testing based on if you want to use the data to train a model or simply inference |


By running `generate_image_dataset.py` the necessary functions will be ran and a new directory called /**data** should exist housing the /**\<type>/target** and /**\<type>/input** datasets with the specified configuration parameters.

Once the image data is created, the synthetic events can be generated. The following parameters can be set according to the use case of the event camera.

| Parameter | Description |
| :--- | :--- |
| `event_sim: Cp` | Log brightness increase constrast threshold for an event trigger|
| `event_sim: Cn` | Log brightness decrease constrast threshold for an event trigger|
| `event_sim: refractory_period` | The minimum time period (us) that a pixel must wait before another event can be triggered |
| `event_sim: sigma_threshold` | Standard deviation of constrast threshold across pixels|
| `event_sim: cutoff_hz` | Frequency of low-pass filter applied to the log intensity changes |
| `event_sim: shot_noise_rate_hz` | Rate of false positive events being triggered simulating sensor noise |
| `event_sim: max_frames` | The number of frames the simulator will generate |
| `event_sim: pause_probability` | The probability that the simulator pauses/skips a period of time |
| `event_sim: rotational_offset` | Addition of constant orientation change in the simulation |
| `event_sim: translational_offset` | Addition of constant translational offset in the simulation |
| `event_sim: max_optical_flow` | Constrains how many pixels a feature can move between frames (lower means smoother sampling)|
| `event_sim: max_interp_frames` | Max number of intermediate frames between keyframes |
| `display` | Boolean to display the raw image and generated events |

To begin the event data generation simply run the ` run_event_simulation.py` script. The event simulation process may take a day or two to complete primarily depending on the number of images being used and the number of frames being generated.


## Training
Once the data set is fully generated, i.e. the target images, input images, and event .dat files are created, then the model training can start. The following parameters can be set prior to training:
| Parameter | Description |
| :--- | :--- |
| `training: type`     | Specify if the data is being used for training or for testing |
| `training: save_dir` | The output directory for the trained models|
| `training: epochs` | The number of passes of the entire dataset through the network|
| `training: batch_size` | The number of examples processed at once before the weights get updated|
| `training: num_workers` | The number of CPU subprocesses to run|
| `training: learning_rate` | The step_size the model takes when updating weights|
| `training: learning_rate_reduction` | The factor to reduce the step size after **learning_rate_epochs** number of epochs|
| `training: learning_rate_epochs` | The number of epochs before the step size gets reduced by a factor of **learning_rate_reduction**|
| `training: lpips_net` | The pre-trained network used to compute the loss [vgg or alex]  |
| `training: num_saves` | The number of model saves throughout the training process|

Once the training parameters are set, the file `train.py` can be ran. This file is located in the **/src** directory.

## Testing & Analysis
### Testing
To test a trained model, you need to download a subset of images that you want to test the model on from the MSCOCO website. Then in the configuration file change `training:type` from `training` to `testing`. Re-run the `generate_image_dataset.py` file and finally run `testing.py` located again in the **/src** directory. This will create a **/results** directory with all of the processed images. 
### Analysis
Once all of the testing images are processed, we can run a performance test. The performance metrics are:
* LPIPS
* PSNR 
* SSIM

Once the performance metrics are generated, they can be compared against papers that use similar pipelines to determine how the model compares. Run `performance_metrics.py` housed in the **/analysis** directory to compute the mean values for all 3 metrics. 
