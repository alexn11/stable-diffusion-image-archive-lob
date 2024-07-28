
import json
import os
import uuid

from PIL import Image

class ImageCache:
    def __init__(self, cache_folder_path: str):
        self.folder = cache_folder_path
        self.meta_data_file = 'meta.json' # TODO use a database
    def prepare_meta_data(self):
        if(not os.path.exists(self.meta_data_file)):
            self.meta_data = {}
        else:
            with open(self.meta_data_file, 'r') as meta_data_file:
                self.meta_data: dict = json.load(meta_data_file)
    def get(self, image_key: str, create_func=None, creat_func_args: dict | None = None) -> Image.Image:
        image_path = self.meta_data.get(image_key)
        if(image_path is None):
            if(create_func is not None):
                return_values = create_func(**creat_func_args)
                self.record(image_key, return_values[0])
                return return_values
            return None
        image_path = self.make_record_file_path(image_path)
        image = Image.open(image_path)
        return image
    def save_meta_data(self,):
        with open(self.meta_data_file, 'w') as meta_data_file:
            json.dump(self.meta_data, meta_data_file)
    def record(self, image_key: str, image: Image.Image):
        image_file_name = uuid.uuid4().hex
        image_file_path = self.make_record_file_path(image_file_name)
        image.save(image_file_path)
        self.meta_data[image_key] = image_file_name
        self.save_meta_data()
    def make_record_file_path(self, file_name: str):
        return os.path.join(self.folder, file_name)
