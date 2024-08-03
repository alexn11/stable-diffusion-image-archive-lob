
import datetime
import json
import os
import uuid

from PIL import Image

def get_utc_now() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y-%m-%d')

class ImageCache:
    def __init__(self, cache_folder_path: str):
        self.folder = cache_folder_path
        self.meta_data_file = os.path.join(self.folder, 'meta.json') # TODO use a database
        self.prepare_meta_data()
    def prepare_meta_data(self):
        if(not os.path.exists(self.meta_data_file)):
            self.meta_data = {}
        else:
            with open(self.meta_data_file, 'r') as meta_data_file:
                self.meta_data: dict = json.load(meta_data_file)
    def get(self,
            image_key: str,
            create_func=None,
            create_func_args: dict | None = None) -> Image.Image | tuple:
        image_data = self.meta_data.get(image_key)
        if(image_data is None):
            if(create_func is not None):
                return_values = create_func(**create_func_args)
                creation_date = get_utc_now()
                self.record(image_key, return_values[0], creation_date)
                return return_values
            return None
        image_path = self.make_record_file_path(image_data[0])
        image = Image.open(image_path)
        return image, image_data[1:]
    def get_image(self, image_key: str, create_func=None, create_func_args: dict | None = None) -> Image.Image:
        return_values = self.get(image_key=image_key,
                                 create_func=create_func,
                                 create_func_args=create_func_args)
        if(isinstance(return_values, tuple)):
            image = return_values[0]
        else:
            image = return_values
        return image
    def save_meta_data(self,):
        with open(self.meta_data_file, 'w') as meta_data_file:
            json.dump(self.meta_data, meta_data_file)
    def record(self, image_key: str, image: Image.Image, creation_date: datetime.datetime | None=None):
        image_file_name = uuid.uuid4().hex + '.png'
        image_file_path = self.make_record_file_path(image_file_name)
        image.save(image_file_path)
        if(creation_date is None):
            creation_date = get_utc_now()
        image_data = (image_file_name, creation_date)
        self.meta_data[image_key] = image_data
        # for easy cleanup:
        #self.meta_data[image_key + '_date_added'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
        self.save_meta_data()
    def make_record_file_path(self, file_name: str):
        return os.path.join(self.folder, file_name)
    def clear_cache(self, threshold_time: datetime.datetime):
        # TODO
        pass
        #
        #self.save_meta_data()
