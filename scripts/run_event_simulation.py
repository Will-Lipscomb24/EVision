import os
import subprocess
import yaml
from pathlib import Path


# --- Configuration ---
# You can use absolute paths or relative paths
BASE_DIR = "/mnt/mnt_pt1/coco4events/data/"
TARGET_DIR = os.path.join(BASE_DIR, "target")
EVENTS_DIR = os.path.join(BASE_DIR, "events")


# Path to the simulator script
# Ensure this path is exactly correct on your system
SIMULATOR_SCRIPT = "/home/wgl294/projects/EVision/tools/openeb/sdk/modules/core_ml/python/samples/viz_video_to_event_simulator/viz_video_to_event_simulator.py"

with open('configs/config.yaml','r') as f:
    config = yaml.safe_load(f)
ev_params = config["event_sim"]


def run_simulation():
    # 1. Create the output directory if it doesn't exist
    os.makedirs(EVENTS_DIR, exist_ok=True)
    print(f"Output directory ready: {EVENTS_DIR}")

    # 2. Loop through IDs 0001 to 1000
    for i, file in enumerate(Path(TARGET_DIR).iterdir(), start=1):
        # Format ID with leading zeros (e.g., 1 -> "0001")
        file_id = f"{i:05d}"
        print(file_id)
        input_img = os.path.join(TARGET_DIR, f"{file_id}.jpg")
        output_dat = os.path.join(EVENTS_DIR, f"{file_id}.dat")

        # 3. Check if input image exists
        if os.path.exists(input_img):
            print("-" * 45)
            print(f"Processing: {input_img}")

            # 4. Construct the command arguments
            cmd = [
                "python", SIMULATOR_SCRIPT,
                str(input_img),
                "-o", str(output_dat),
                "--no_display",
                "--Cp", str(ev_params['Cp']),
                "--Cn", str(ev_params['Cn']),
                "--refractory_period", str(ev_params['refractory_period']),
                "--sigma_threshold", str(ev_params['sigma_threshold']),
                "--cutoff_hz", str(ev_params['cutoff_hz']),
                "--shot_noise_rate_hz", str(ev_params['shot_noise_rate_hz']),
                "--set_frames", str(ev_params['set_frames']),
                "--pause_probability", str(ev_params['pause_probability'])
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