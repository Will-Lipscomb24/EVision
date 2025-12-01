# BLENDER RENDER SETTINGS 
import bpy #File will be run pointing to blender so local installation of bpy is not needed
import os
import yaml
import random
from mathutils import Vector, Matrix

''' New Method:
Have a stationary scene with a moving camera. Use a few path generations for the camera to reference and use a noise function to add noise to each path for uniqueness.
I will also need to remove and add objects to the position of the parent objects and scale them to a reasonable value. Then I can create frames rendered at an animation FPS of >= 24 fps for smooth stitching.
I may also want to add random backgrounds to add more diversity to the areas around the main objects (planet and spacecraft). 
'''

# ---- Load YAML ----
with open("/home/will/projects/EVision/configs/render_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cam_params = cfg['cam_params']
render_params = cfg['render_params']
output_params = cfg['output_params']
scene_data = cfg['scene_data']

# ---- Initialize Scene Objects ----
scene = bpy.context.scene
cam = bpy.data.objects['Camera']
soho = bpy.data.objects['Soho']
earth = bpy.data.objects['Earth']
sun = bpy.data.objects['Sun']
lamp = bpy.data.objects['NICELAMP']
small_sun = bpy.data.collections['Sun Glare']

# ---- Configure Render Settings ----
scene.frame_start = 1
frames_per_run = render_params['frames_per_animation']
scene.frame_end = frames_per_run
num_runs = render_params['num_animations']
scene.render.resolution_x = cam_params['resolution_x']
scene.render.resolution_y = cam_params['resolution_y']
frame_start = scene.frame_start
frame_end   = scene.frame_end
scene.render.fps = cam_params['fps']
scene.render.image_settings.file_format = 'PNG'  # <-- Add this

# ---- Camera intrinsics ----
cam.data.lens = cam_params['focal_length']              
cam.data.sensor_width = cam_params['sensor_width']
cam.data.sensor_height = cam_params['sensor_height']
cam.data.sensor_fit = cam_params['sensor_fit']

    # ---- Function to Set Initial Orbit Pose ----
def set_orbit_pose():
    earth.sssim_orbit.time_offset = random.uniform(0.0, 1.0)
    soho.sssim_orbit.time_offset = random.uniform(0.0, 1.0)
    soho.sssim_orbit.distance_mantissa = random.randint(5000, 10000)
    soho.sssim_orbit.distance_exp = 1
    soho.sssim_rotation.axis_tilt = random.uniform(0,360)
    soho.sssim_rotation.tilt_direction = random.uniform(0,360)
    earth.sssim_rotation.axis_tilt = random.uniform(0,180)
    earth.sssim_rotation.tilt_direction = random.uniform(0,180)
    bpy.ops.scene.update_sssim_drivers()
    
    # ---- Function to Set Initial Camera Pose ----
def set_camera_pose():
    soho_world = soho.matrix_world.translation
    rel_soho_to_earth = earth.location - soho_world
    cam_offset = random.uniform(0.05,.35)
    cam.location = soho_world - cam_offset*rel_soho_to_earth
    direction = soho_world - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam_to_target = (soho_world - cam.location).normalized()
    return cam_to_target
    

  
# ---- Function to set ideal lighting ----
def set_ideal_lighting(cam_to_target):
    sun.hide_render = True
    small_sun.hide_render = True 
    lamp.location = cam.location - cam_to_target * 0.5
    lamp.rotation_euler = cam.rotation_euler
    lamp.data.energy = 20

## ---- Directories that Hou'se Random .blend Files ----
#planet_dir = '/home/will/projects/EVision/assets/planet_blend_files'
#spacecraft_dir = '/home/will/projects/EVision/assets/spacecraft_blend_files'
#def set_objects();
#    planet_file = random.choice([f for f in os.listdir(planet_dir) if f.endswith(".blend")])
#    spacecraft_file = random.choice([f for f in os.listdir(spacecraft_dir) if f.endswith(".blend")])

#    planet_path = os.path.join(planet_dir, planet_file)
#    spacecraft_path = os.path.join(spacecraft_dir, spacecraft_file)

    


# ---- Directory to save frames ----
ideal_lighting_dir  = "/home/will/projects/EVision/data/render_output/ideal_lighting/"
os.makedirs(ideal_lighting_dir, exist_ok=True)

## ---- FFmpeg helper ----
#def make_video(input_folder, output_video, fps):
#    cmd = [
#        "ffmpeg", "-y",
#        "-framerate", str(fps),
#        "-i", os.path.join(input_folder, "frame_%04d.png"),
#        "-c:v", "libx264",
#        "-pix_fmt", "yuv420p",
#        output_video
#    ]
#    subprocess.run(cmd, check=True)

# ---- Main Loop ----

for run_idx in range(num_runs):
    print(f"=== Run {run_idx+1} ===")
    set_orbit_pose()

    run_dir = os.path.join(ideal_lighting_dir,f'run_{run_idx:04d}')
    os.makedirs(run_dir,exist_ok=True)

    for frame_offset in range(frames_per_run):
        frame = frame_offset + 1
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        cam_to_target = set_camera_pose()
        set_ideal_lighting(cam_to_target)
        bpy.context.scene.render.filepath = os.path.join(run_dir, f"frame_{frame:04d}.png")
        bpy.ops.render.render(animation=True)
        print(f'=== Frame {frame} Generated ===')
        
#      # ---- Create videos ----
#    random_video_out = os.path.join(harsh_lighting_dir, f"run_{run_idx+1}.mp4")
#    ideal_video_out  = os.path.join(ideal_lighting_dir,  f"run_{run_idx+1}.mp4")

#    print(f"[FFMPEG] Encoding random lighting video for run {run_idx+1}")
#    make_video(random_folder, random_video_out, scene.render.fps)

#    print(f"[FFMPEG] Encoding ideal lighting video for run {run_idx+1}")
#    make_video(ideal_folder, ideal_video_out, scene.render.fps)