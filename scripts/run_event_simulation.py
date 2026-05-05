import os
import glob
import subprocess
import yaml
from src.utils import find_evision_root

REPO_ROOT = find_evision_root()
configs_path = REPO_ROOT / 'configs' / 'config.yaml'
# --- Configuration ---
with open(configs_path, 'r') as f:
    config = yaml.safe_load(f)
ev_params = config["event_sim"]

type = config['training']['type']
TARGET_DIR = str(REPO_ROOT / 'data' / f'{type}' / 'target')
EVENTS_DIR = str(REPO_ROOT / 'data' / f'{type}' / 'events')

# Path to the simulator script
# Ensure this path is exactly correct on your system
SIMULATOR_SCRIPT = REPO_ROOT / "tools" / "openeb" / "sdk" / "modules" / "core_ml" / "python" / "samples" / "viz_video_to_event_simulator" / "viz_video_to_event_simulator.py"

image_paths = sorted(glob.glob(os.path.join(TARGET_DIR, "*.jpg")))


def run_simulation():
    # 1. Create the output directory if it doesn't exist
    os.makedirs(EVENTS_DIR, exist_ok=True)
    print(f"Output directory ready: {EVENTS_DIR}")

    # 2. Loop through image ID's
    for img_path in image_paths:
        
        file_id = os.path.splitext(os.path.basename(img_path))[0]

        input_img = img_path
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
                "--Cp", str(ev_params['Cp']),
                "--Cn", str(ev_params['Cn']),
                "--refractory_period", str(ev_params['refractory_period']),
                "--sigma_threshold", str(ev_params['sigma_threshold']),
                "--cutoff_hz", str(ev_params['cutoff_hz']),
                "--shot_noise_rate_hz", str(ev_params['shot_noise_rate_hz']),
                "--max_frames", str(ev_params['max_frames']),
                "--pause_probability", str(ev_params['pause_probability']),
                "--rotational_offset", str(ev_params['rotational_offset']),
                "--translational_offset", str(ev_params['translational_offset'])
            ]
            if ev_params['display']:
                cmd.append("--display")
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
