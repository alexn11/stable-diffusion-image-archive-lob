
import json
import os
import uuid

from PIL import Image

class ImageCache:
    def __init__(self, cache_folder_path: str):
        self.folder = cache_folder_path
        self.meta_data_file = 'meta.json' # TODO use a database
    def prepare_meta_data(self):
        with open(self.meta_data_file, 'r') as meta_data_file:
            self.meta_data: dict = json.load(meta_data_file)
    def get(self, image_key: str) -> Image.Image:
        image_path = self.meta_data.get(image_key)
        if(image_path is None):
            return None
        image_path = self.make_record_file_path(image_path)
        image = Image.open(image_path)
        return image
    def record(self, image_key: str, image: Image.Image):
        image_file_name = uuid.uuid4().hex
        image_file_path = self.make_record_file_path(image_file_name)
        image.save(image_file_path)
        self.meta_data[image_key] = image_file_name
    def make_record_file_path(self, file_name: str):
        return os.path.join(self.folder, file_name)
