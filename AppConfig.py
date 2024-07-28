import os

from dotenv import load_dotenv

from model_constants import image_height, image_width

class AppConfig:
    def __init__(self, dotenv_file='api.env'):
        load_dotenv(dotenv_file)
        self.image_cache_folder = os.environ.get('IMAGE_CACHE_FOLDER', '.')
        self.make_image_finder_config_dict()
    def make_image_finder_config_dict(self,):
        seed = os.environ.get('SEED', '0')
        debug = os.environ.get('DEBUG', '')
        self.image_finder_config_dict = {
            'seed': int(seed),
            'debug': (debug in ['1', 'True', 'true', 'on']),
            'model_name': os.environ.get('MODEL_NAME', 'stabilityai/stable-diffusion-2-1-unclip-small'),
            'device': os.environ.get('DEVICE', 'cuda'),
            'prompt': '',
            'num_inference_steps': int(os.environ.get('NUM_INFERENCE_STEPS', 50)),
            'height': image_height,
            'width': image_width,
            'batch_size': 1,
            'num_images_per_prompt': 1,
            'guidance_scale': 7.5,
            'output_type': 'pil',
        }
