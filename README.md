# EVision
This repository will serve as a pipeline for image reconstuction using a RGB/Event Camera data fusion.

Dataset:
The dataset used for the training & validation pipelines is the MSCOCO 2017 train images(118K/18GB) and 2017 val images(5K/1GB).
Only a subset of the full training dataset was used (~40K).

### A. 
To extract a subset of the dataset into a new dir run: 

`mkdir image_subset`

`ls | shuf -n 20000 | xargs -I{} cp "{}" image_subset/`

Dataset Manipulation:
There are 2 ways in which we will modify the original data:
1. Using the ground truth imagery to generate small event-camera clips using Metavision OpenEB tools.
2. Over-exposing the ground truth images to generate the models input image.



1
In order to generate events from images, we will use OpenEB planar renderer. To install OpenEB,
follow the guideline here: https://github.com/prophesee-ai/openeb






Working with Docker Container:
AA. To initialize a container from an image run: 
				docker run -it \
					-e DISPLAY=$DISPLAY \
					-v /tmp/.X11-unix:/tmp/.X11-unix \
					-v <path to folder on local machine>:<path to folder in container> \
					<image name> 
This script will allow for docker to use your hosts displays and it will allow you to mount a specified 
folder from your host.

A. To start a created container run:
				docker start -ai <container name> 

B. To open an existing container run:
				docker exec -it <container name> bash

C. If you want to use your host's display in the container:
	On host run:	
				xhost +local:root

D. To set the environment variables in the esim docker container run:
				source /root/sim_ws/devel/setup.bash
*Note that this can be added to your .bashrc script to run on startup





7
RUNNING THE SIMULATOR:
USING ESIM ROS:
	A. Navigate to the event simulator by running:
					roscd esim_ros 
	B. Run the simulator on multiple images within a folder by running:
					./run_all_textures.sh
	*Within the folder you will need to specify the path to the folder that contains all of the imagery, 
	as well as the folder that you want the .bag files to go into

	C. To visualize the .bag output of the simulator you will need to have 3 ros terminals open. On terminal (1) run:
					roscore 
	or have the same terminal open that you used to generate the .bag files
	On (2) run: 
					roscd esim_visualization
	ON (3) navigate to the .bag file that you want to visualize and run:
					rosbag play <.bag file>
		

	D. These are the topics saved in the rosbag after running the simulation:
		topics:  /cam0/camera_info         20 msgs    : sensor_msgs/CameraInfo    
				/cam0/depthmap            20 msgs    : sensor_msgs/Image         
				/cam0/events             113 msgs    : dvs_msgs/EventArray       
				/cam0/image_corrupted     56 msgs    : sensor_msgs/Image         
				/cam0/image_raw           57 msgs    : sensor_msgs/Image         
				/cam0/optic_flow          20 msgs    : esim_msgs/OpticFlow       
				/cam0/pointcloud          20 msgs    : sensor_msgs/PointCloud2   
				/cam0/pose              2413 msgs    : geometry_msgs/PoseStamped 
				/cam0/twist             2413 msgs    : geometry_msgs/TwistStamped
				/imu                    2000 msgs    : sensor_msgs/Imu           
				/tf                     2413 msgs    : tf/tfMessage
	For our model, we need to save:
		image_raw
		pointcloud
		events
USING METAVISION SCRIPTS
	A. OpenEB will need to be installed and built. You can do so via this Github repo: https://github.com/prophesee-ai/openeb
	B. Once cloned, the path to the event simulator is located here:
		<OpenEB repo>/sdk/modules/core_ml/python/samples/viz_video_to_event_simulator/viz_video_to_event_simulator.py
	C. The simulator has the following args which let you adjust the output of the event simulator:
		def parse_args(only_default_values=False):
		parser = argparse.ArgumentParser(description='Run a simple event based simulator on a video or an image',
										formatter_class=argparse.ArgumentDefaultsHelpFormatter)

		parser.add_argument('path', help='path to a video or an image from which we will produce'
							' the corresponding events ')

		parser.add_argument('--n_events', type=int, default=50000,
							help='number of events to display at once')
		parser.add_argument('--height_width', nargs=2, default=None, type=int,
							help="if set, scales the input image to the requested values.")
		parser.add_argument('--crop_image', action="store_true", help='crop images instead of resizing them.')
		parser.add_argument("--no_display", dest="display", action="store_false", help='disable the graphical return.')
		parser.add_argument("--verbose", action="store_true", help='set to have the speed of the simulator in ev/s')
		parser.add_argument('-o', "--output", help="if provided, will write the events in a DAT file in the corresponding path")
		parser.add_argument('-fps', '--override_fps', default=0, type=float,
							help="if positive, overrides the framerate of the input video. Useful for slow motion videos.")

		simulator_options = parser.add_argument_group('Simulator parameters')
		simulator_options.add_argument("--Cp", default="0.05", type=float,
									help="mean for positive event contrast threshold distribution")
		simulator_options.add_argument("--Cn", default="0.05", type=float,
									help="mean value for negative event contrast threshold distribution")
		simulator_options.add_argument(
			"--refractory_period", default=1, type=float,
			help="time interval (in us), after firing an event during which a pixel won't emit a new event.")
		simulator_options.add_argument(
			"--sigma_threshold", type=float, default="0.03", help="standard deviation for threshold"
			"distribution across the array of pixels. The higher it is the less reliable the imager.")
		simulator_options.add_argument("--cutoff_hz", default=200, type=float,
									help="cutoff frequency for photodiode latency simulation")
		simulator_options.add_argument("--leak_rate_hz", type=float, default=0,
									help="frequency of reference value leakage")
		simulator_options.add_argument("--shot_noise_rate_hz", default=10, type=float,
									help="frequency for shot noise events")

		return parser.parse_args()

	D. Here is sample code to run the simulator:
		python /home/will/projects/EVision/tools/ev_camera/openeb/sdk/modules/core_ml/python/samples/viz_video_to_event_simulator/viz_video_to_event_simulator.py 
		/home/will/projects/EVision/data/over_exposed/image_2_o_exp3.47_gam3.41.jpg 
		--Cp 0.02 --Cn 0.02 --refractory_period 100 --sigma_threshold .04 --cutoff_hz 500 --shot_noise_rate_hz 10 
