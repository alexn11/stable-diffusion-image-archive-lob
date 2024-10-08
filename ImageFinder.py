# Copyright 2024 Alexandre De Zotti. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from diffusers.utils.testing_utils import enable_full_determinism
from PIL import Image
import torch

from prepare_model import prepare_config, prepare_model
from key_to_image_pipeline import key_to_image
from prompt_to_key import generate_key_from_prompt

class ImageFinder:
    def __init__(self, config: dict, debug=False):
        self.debug = debug
        self.config = config
        self.max_steps = config['max_steps']
        self.model_config = prepare_config(model_name=config['model_name'],
                                      device=config['device'],
                                      num_images_per_prompt=config['num_images_per_prompt'],
                                      height=config['height'],
                                      width=config['width'],
                                      batch_size=config['batch_size'],
                                      guidance_scale=config['guidance_scale'],)
        enable_full_determinism()
        self.generator = None
        self.device = self.model_config['device']
        self.dtype = self.model_config['dtype']
        self.num_inference_steps = self.model_config['num_inference_steps']
        self.num_images_per_prompt = self.model_config['num_images_per_prompt']
        self.batch_size = self.model_config['batch_size']
        self.image_height = self.model_config['height']
        self.image_width = self.model_config['width']
        self.guidance_scale = self.model_config['guidance_scale']
        self.do_classifier_free_guidance = self.model_config['do_classifier_free_guidance']
        self.pipe_generator = torch.Generator(device=self.device).manual_seed(self.config['seed'])
        #pipe_generator = None
        self.pipe = prepare_model(self.model_config['model_name'],
                                  self.dtype,
                                  self.device)
    def find(self, key: str, return_tuple=True) -> Image.Image:
        image = key_to_image(key,
                     pipe=self.pipe,
                     generator=self.pipe_generator,
                     dtype=self.dtype,
                     device=self.device,
                     num_inference_steps=self.num_inference_steps,
                     num_images_per_prompt=self.num_images_per_prompt,
                     batch_size=self.batch_size,
                     height=self.image_height,
                     width=self.image_width,
                     guidance_scale=self.guidance_scale,
                     do_classifier_free_guidance=self.do_classifier_free_guidance,
                     max_steps=self.max_steps,
                     debug=self.debug)
        if(return_tuple):
            return (image, )
        return image
    def find_a_key(self, prompt: str) -> str:
        key = generate_key_from_prompt(prompt=prompt,
                                       pipe=self.pipe,
                                       device=self.device,
                                       num_images_per_prompt=self.num_images_per_prompt,
                                       latents=None,)
        return key
    def find_from_prompt(self, prompt: str) -> tuple[Image.Image, str]:
        key = self.find_a_key(prompt)
        image = self.find(key)
        return (image, key, )