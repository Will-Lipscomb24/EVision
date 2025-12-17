#!/bin/bash

# Loop through all 1000 directories
for i in {0001..1000}; do
    folder="../data/all_data/image_$i"
    input_img="$folder/image_${i}_o.jpg"
    output_dat="$folder/events_${i}.dat"

    # Check if the input image exists before running
    if [ -f "$input_img" ]; then
        echo "--------------------------------------------"
        echo "Processing: $input_img"
        
        # Run the simulator
        # --no_display: hides the window so it runs faster
        # -o: specifies the output path for the .dat file
        python /home/will/projects/EVision/tools/ev_camera/openeb/sdk/modules/core_ml/python/samples/viz_video_to_event_simulator/viz_video_to_event_simulator.py \
        "$input_img" \
   	-o "$output_dat" \
    	--no_display \
   	--Cp 0.05 \
   	--Cn 0.05 \
    	--refractory_period 1 \
    	--sigma_threshold .001 \
    	--cutoff_hz 0 \
    	--shot_noise_rate_hz 10
        echo "Saved to: $output_dat"
    else
        echo "Skipping $folder: $input_img not found."
    fi
done

echo "Done! All simulations processed."
