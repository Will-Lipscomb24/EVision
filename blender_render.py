# BLENDER RENDER SETTINGS 
import bpy #File will be run pointing to blender so local installation of bpy is not needed
import os
import yaml
import random
from mathutils import Vector, Matrix

# ---- Load YAML ----
with open("/home/will/projects/EVision/settings/render_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cam_params = cfg['cam_params']
render_params = cfg['render_params']
output_params = cfg['output_params']
scene_data = cfg['scene_data']

# ---- Initialize Scene Objects ----
scene = bpy.context.scene
cam = bpy.data.objects['Camera']
soho = bpy.data.objects['Soho']
soho_mat = bpy.data.objects['soho_instruments']
earth = bpy.data.objects['Earth']
sun_mat = bpy.data.objects['Sun_Surface']
sun = bpy.data.objects['Sun']
lamp = bpy.data.objects['NICELAMP']

# ---- Configure Render Settings ----
scene.frame_start = 1
frames_per_run = render_params['frames_per_animation']
scene.frame_end = frames_per_run
num_runs = render_params['num_animations']
scene.render.resolution_x = cam_params['resolution_x']
scene.render.resolution_y = cam_params['resolution_y']
frame_start = scene.frame_start
frame_end   = scene.frame_end
scene.frame_rate = render_params['fps']

# ---- Camera intrinsics ----
cam.data.lens = cam_params['focal_length']              
cam.data.sensor_width = cam_params['sensor_width']
cam.data.sensor_height = cam_params['sensor_height']
cam.data.sensor_fit = cam_params['sensor_fit']

# ---- Material Initializing ----
sun_material = sun_mat.active_material
soho_material = soho_mat.active_material
sun_nodes = sun_material.node_tree.nodes
soho_nodes = soho_material.node_tree.nodes
soho_bsdf_node = soho_nodes[0]
sun_emission_node = sun_nodes[1]


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
    cam_offset = random.uniform(0.3,.75)
    cam.location = soho_world - cam_offset*rel_soho_to_earth
    direction = soho_world - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam_to_target = (soho_world - cam.location).normalized()
    return cam_to_target
    
    # ---- Function to set random lighting ----
def set_random_lighting(cam_to_target):
    sun_emission_node.inputs['Strength'].default_value = random.uniform(500000, 750000)
    soho_bsdf_node.inputs['Metallic'].default_value = random.uniform(0.1, 1)
    soho_bsdf_node.inputs['Roughness'].default_value = random.uniform(0.1, 1)
    lamp.location = cam.location - cam_to_target * 0.5
    lamp.rotation_euler = cam.rotation_euler
    sun.hide_render = False

# ---- Function to set ideal lighting ----
def set_ideal_lighting(cam_to_target):
    sun.hide_render = True
    sun_emission_node.inputs['Strength'].default_value = 125000
    soho_bsdf_node.inputs['Metallic'].default_value = 0.5
    soho_bsdf_node.inputs['Roughness'].default_value = 0.5
    lamp.location = cam.location - cam_to_target * 0.5
    lamp.rotation_euler = cam.rotation_euler


# ---- Directory to save frames ----
harsh_lighting_dir = "/home/will/projects/EVision/data/render_output/harsh_lighting/"
ideal_lighting_dir  = "/home/will/projects/EVision/data/render_output/ideal_lighting/"

os.makedirs(harsh_lighting_dir, exist_ok=True)
os.makedirs(ideal_lighting_dir, exist_ok=True)


# ---- Main Loop ----
objects_to_lock = [soho,earth,cam,lamp]
for run_idx in range(num_runs):
    print(f"=== Run {run_idx+1} ===")
    
    # Create folders for this run
    random_folder = os.path.join(harsh_lighting_dir, f"run_{run_idx+1}/random")
    ideal_folder  = os.path.join(ideal_lighting_dir, f"run_{run_idx+1}/ideal")
    os.makedirs(random_folder, exist_ok=True)
    os.makedirs(ideal_folder, exist_ok=True)
    
    set_orbit_pose()
    cam_to_target = set_camera_pose()
    
    # ---- Save initial transforms ----
    initial_transforms = {}
    for obj in objects_to_lock:
        initial_transforms[obj.name] = {
            'location': obj.location.copy(),
            'rotation': obj.rotation_euler.copy(),
            'scale': obj.scale.copy()
        }

    # ---- Random lighting run ----
    set_random_lighting(cam_to_target)
    for frame_offset in range(frames_per_run):
        frame = frame_offset + 1
        bpy.context.scene.frame_set(frame)
        bpy.context.scene.render.filepath = os.path.join(random_folder, f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        
    for obj in objects_to_lock:
        obj.location = initial_transforms[obj.name]['location']
        obj.rotation_euler = initial_transforms[obj.name]['rotation']
        obj.scale = initial_transforms[obj.name]['scale']

    # ---- Ideal lighting run (same frames) ----
    set_ideal_lighting(cam_to_target)
    for frame_offset in range(frames_per_run):
        frame = frame_offset + 1
        bpy.context.scene.frame_set(frame)
        bpy.context.scene.render.filepath = os.path.join(ideal_folder, f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)