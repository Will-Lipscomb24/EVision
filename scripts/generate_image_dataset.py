from src.utils import resize_images, random_select_and_rename, convert_images_to_grayscale,find_evision_root
from scripts.expose import process_full_dataset
import os 
import yaml 

REPO_ROOT = find_evision_root()

data_path = '/home/will/Downloads/test2017'
configs_dir = REPO_ROOT / 'configs' / 'config.yaml'


with open(configs_dir, 'r') as f:
    config = yaml.safe_load(f)

desired_images = config['model']['desired_images']
new_height = config['model']['height']
new_width = config['model']['width']
type = config['training']['type']

target_dir = str(REPO_ROOT / 'data' / f'{type}' / 'target')
input_dir = str(REPO_ROOT / 'data' / f'{type}' / 'input')

os.makedirs(target_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

# Create target image directory
random_select_and_rename(data_path, target_dir, target_count=desired_images, start_index=1)
resize_images(target_dir, target_dir, new_height, new_width)
convert_images_to_grayscale(target_dir, target_dir)

# Create poorly exposed image directory
process_full_dataset(target_dir, input_dir)
