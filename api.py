import base64
import io

from fastapi import FastAPI, Depends
from pydantic import BaseModel, constr, validator

from model_constants import key_length
from key_strings import base64_characters
from key_strings import generate_random_key_base64, get_next_key

from AppConfig import AppConfig
from ImageCache import ImageCache
from ImageFinder import ImageFinder

class Key(BaseModel):
    key: str

class Prompt(BaseModel):
    prompt: str

# i would put the below into the above if i understood how this worked...
def validate_key(key):
    key = ''.join(list(filter(lambda c: c in base64_characters, key)))
    if(len(key) != key_length):
        this_len = len(key)
        if(this_len > key_length):
            key = key[:key_length]
        else:
            padding_len = key_length - this_len
            key += padding_len * 'A'
    return key
# key: str = constr(pattern=rf'[{base64_characters}]{{{2}}}$')


class ImageFinderPipeline:
    def __init__(self,):
        self.app_config = AppConfig()
        self.image_cache = ImageCache(self.app_config.image_cache_folder)
        self.image_finder = ImageFinder(self.app_config.image_finder_config_dict)


def get_image(key: str | None = None,
              prompt: str | None = None,
              image_finder_pipeline: ImageFinderPipeline = None,):
    if((key is None) and (prompt is not None)):
        key = image_finder_pipeline.image_finder.find_a_key(prompt)
    if(key is not None):
        key = validate_key(key)
        image = image_finder_pipeline.image_cache.get_image(key,
                                                            create_func=image_finder_pipeline.image_finder.find,
                                                            create_func_args={'key': key})
        image_byte_stream = io.BytesIO()
        image.save(image_byte_stream, format="JPEG")
        encoded_image_base64 = base64.b64encode(image_byte_stream.getvalue()).decode("utf-8")
    if(prompt is None):
        return encoded_image_base64
    return encoded_image_base64, key

def get_image_finder():
    return ImageFinderPipeline()

def get_next_image(key: str, image_finder: ImageFinderPipeline, direction=1):
    key = validate_key(key)
    next_key = get_next_key(key, direction=direction)
    encoded_image_base64 = get_image(key=next_key, image_finder_pipeline=image_finder)
    return {
        'key': next_key,
        'image': encoded_image_base64,
    }

app = FastAPI()

@app.get('/image/random')
def random_image(image_finder: ImageFinderPipeline = Depends(get_image_finder)):
    key = generate_random_key_base64()
    encoded_image_base64 = get_image(key=key, image_finder_pipeline=image_finder)
    return {
        'key': key,
        'image': encoded_image_base64,
    }

@app.get('/key/random')
def random_key():
    key = generate_random_key_base64()
    return {
        'key': key,
    }

@app.post('/image/next')
def next_image(key: Key, image_finder: ImageFinderPipeline = Depends(get_image_finder)):
    return get_next_image(key.key, image_finder, direction=1)

@app.post('/image/prev')
def next_image(key: Key, image_finder: ImageFinderPipeline = Depends(get_image_finder)):
    return get_next_image(key.key, image_finder, direction=-1)

@app.post('/image')
def key_to_image(key: Key,
                 image_finder: ImageFinderPipeline = Depends(get_image_finder)):
    key = validate_key(key.key)
    encoded_image_base64 = get_image(key=key, image_finder_pipeline=image_finder)
    return {
        'key': key,
        'image': encoded_image_base64,
    }

@app.post('/image/by-prompt')
def promt_to_image(prompt: Prompt,
                   image_finder: ImageFinderPipeline = Depends(get_image_finder)):
    encoded_image_base6, image_key = get_image(prompt=prompt.prompt, image_finder_pipeline=image_finder)
    return {
        'key': image_key,
        'image': encoded_image_base6,
    }
