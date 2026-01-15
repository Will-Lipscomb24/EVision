import os
import subprocess
import yaml


# --- Configuration ---
# You can use absolute paths or relative paths
BASE_DIR = "./data/data_unformatted"
INPUT_DIR = os.path.join(BASE_DIR, "target")
EVENTS_DIR = os.path.join(BASE_DIR, "events")


# Path to the simulator script
# Ensure this path is exactly correct on your system
SIMULATOR_SCRIPT = "/home/will/projects/EVision/tools/openeb/sdk/modules/core_ml/python/samples/viz_video_to_event_simulator/viz_video_to_event_simulator.py"
GPU_SIMULATOR_SCRIPT = "/home/will/projects/EVision/tools/openeb/sdk/modules/core_ml/python/samples/viz_video_to_event_gpu_simulator/viz_video_to_event_gpu_simulator.py"
GPU_SIM = "/home/will/projects/EVision/scripts/viz_video_to_event_gpu_simulator_w_motion.py"
with open('configs/config.yaml','r') as f:
    config = yaml.safe_load(f)

ev_config = config['event_sim']
ev_params = config["event_sim"]['cpu']
ev_params_gpu = config['event_sim']['gpu']
mp = config['model']

def run_simulation():
    # 1. Create the output directory if it doesn't exist
    os.makedirs(EVENTS_DIR, exist_ok=True)
    print(f"Output directory ready: {EVENTS_DIR}")

    # 2. Loop through IDs 0001 to 1000
    for i in range(1, 1001):
        # Format ID with leading zeros (e.g., 1 -> "0001")
        file_id = f"{i:04d}"
        print(file_id)
        input_img = os.path.join(INPUT_DIR, f"{file_id}.jpg")
        output_dat = os.path.join(EVENTS_DIR, f"{file_id}.dat")

        # 3. Check if input image exists
        if os.path.exists(input_img):
            print("-" * 45)
            print(f"Processing: {input_img}")

            # 4. Construct the command arguments
            cmd = [
                "python", GPU_SIMULATOR_SCRIPT,
                str(input_img),
                "--threshold-mu", str(ev_params_gpu['mean_mu']),
                "--threshold-std", str(ev_params_gpu['std_mu']),
                "--refractory-period", str(ev_params['refractory_period']),
                "--cutoff-hz", str(ev_params['cutoff_hz']),
                "--leak-rate-hz", str(ev_params['leak_rate_hz']),
                "--shot-noise-hz", str(ev_params['shot_noise_rate_hz']),
                "--batch-size", str(ev_params_gpu['batch_size']),
                "--nbins",str(ev_params_gpu['voxel_bins']),
                "--height", str(mp['height']),
                "--width", str(mp['width'])
            ]

            try:
                # 5. Run the command
                # capture_output=True keeps your terminal clean, remove it to see simulator logs
                subprocess.run(cmd, check=True) 
                print(f"Saved to: {output_dat}")
                
            except subprocess.CalledProcessError as e:
                print(f"ERROR processing {file_id}: {e}")
        
        else:
            # Optional: Uncomment if you want to see what's missing
            # print(f"Skipping {file_id}: {input_img} not found.")
            pass

    print("Done! All simulations processed.")

if __name__ == "__main__":
    run_simulation()