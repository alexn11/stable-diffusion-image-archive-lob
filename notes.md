# normal use

```bash
python key_to_image.py
# 
python key_to_image.py --prompt "a dog" --num-inference-steps 40 --nb-keys 4
python key_to_image.py --prompt "a nice ice cream" --num-inference-steps 36 --nb-keys 4
python key_to_image.py --num-inference-steps 12 --nb-keys 4
python key_to_image.py --key-file test-key.txt --num-inference-steps 8
python key_to_image.py --key-file test-key.txt --num-inference-steps 8 --check-determinism
python key_to_image.py --key-file test-key.txt --num-inference-steps 26
python key_to_image.py --nb-keys 4 --num-inference-steps 26
python key_to_image.py --nb-keys 4 --num-inference-steps 12 --latents-type fixed-generator
python key_to_image.py --key-file test-key.txt --num-inference-steps 8 --check-determinism --latents-type fixed-generator
python key_to_image.py --nb-keys 4 --num-inference-steps 12 --latents-type blob
ython key_to_image.py --key-file test-key.txt --num-inference-steps 8 --latents-type fixed-generator --output test3
```



## old version pre 2024-07-23
```bash
python key_to_image.py
python key_to_image.py --prompt "a dog" --num-inference-steps 40 --nb-keys 4
python key_to_image.py --prompt "a nice ice cream" --num-inference-steps 36 --nb-keys 4
python key_to_image.py --num-inference-steps 12 --nb-keys 4
python key_to_image.py --key-file test-key.txt --num-inference-steps 8
python key_to_image.py --key-file test-key.txt --num-inference-steps 8 --check-determinism
python key_to_image.py --key-file test-key.txt --num-inference-steps 26
python key_to_image.py --nb-keys 4 --num-inference-steps 26
python key_to_image.py --nb-keys 4 --num-inference-steps 12 --latents-type fixed-generator
python key_to_image.py --key-file test-key.txt --num-inference-steps 8 --check-determinism --latents-type fixed-generator
python key_to_image.py --nb-keys 4 --num-inference-steps 12 --latents-type blob
ython key_to_image.py --key-file test-key.txt --num-inference-steps 8 --latents-type fixed-generator --output test3
```

# ideas

- latents shape is rather small might be worth using this as a key; add it to the key mayeb?

- the only way to ensure reproducibility is to create the pipe from scratch every time an image is generated

- use a small resolution model
- use a quantised model
- use the embedding of the prompt as key
- what is the set of images accessible to the model?
- is there a mathematical characterisation of images accessible to diffusion models?
- how many steps to use?
- key -> image
- inverse problem (image to key): use an optimisation approach?
- inverse problem (image to key): can this be done without optimisation?
- choice of the initial image?
- choice of the initial image: all black, series of pure colour pixels, some fixed image
- choice of the initial image: dependend on the key?

# todos

- rewrite the genration loop (its copy paste from diffusers)
- put it in a function
- image is not related to prompt ?
- cant decrease value of latents without loosing relevant bits (maybe reorganise key system)
- prompt still doesnt work
- so far generated images are a bit boring
- convert text prompt to key :)
- 120 inference steps is a good number
- 76 ?

## done

- in normal key latents should have different exponent bias (-14 to 1 instaed of -12 to 3)
- promtp kindof work but its not a dog?
- new bug unlocked: random exactly 2 difference in exactly one component between orig vector and its re-conversion
  specifically (seems to) happen(s) when the original data is 0. which gets converted into 2. (one bit bug?)
  bad solution: replace 0. wuth the smallest number possible
  actually its not a bug its a feature (i remember now)
  so yeah use bad solution
- sorted annoying binary conversion functions bugs
- there is an issue with the last bits in the key
- there is an issue with key length
- unpack might be bugged stupid me believing chatgpt
- initial image: set it up
- clip exponent (not too low)
- try better addresses than just digits

# notes


## modèle
- babelia resolution: 640x416


## tech

- [https://keras.io/examples/generative/random_walks_with_stable_diffusion/]
- stable diffusion prompt encoding shape: (77, 768), dim total: 59136
- the embedding computed in the hugging face pipe is actually a duplication of the above embedding, hence has shape (2,77,768) but with twice the same vector
- size of embedding vector, in bytes: 2*77*768 = 118272 (times 2 is for float16 used only with gpu, on cpu would be double)
    - bits: 946176 = 16*77*768
    - in base ten: 10^283851  (283851=3*94617) times 64 times 1.024**94617
    - 1.024^94617 = (1.024^23654)^2 (=exp is 47308) ^2 (=exp is 94616) times 1.024
    - 1.024^23654 = 4.316923554669762e+243
    - (10^243)^2^2 = 10^972
    - 1.024*((4.316923554669762)**4) = 355.6291805745891
    - 1.024^94617 = 3.556291805745891 times 10^974
    - final result: 3.556291805745891 times 10^284825 (284825 = 283851 + 974)
    - needs 284826 digits with some forbidden values (that's an improvement compare to one million)

## float format

- float 16 bits: 15=sign, 14-10 (5bites) = exponent, 10: (10 bits) fraction
- exponent format is 00001 -> -14 to 11110 -> 15, with special meaning for exp 00000 (zero if fract is zero) & 11111 (infty, nan)
- 12.0: 12 in binary: 0b1100
- 12.0 in float 16: 0b0100101000000000
- two bytes:  [ 0b01001010, 0 ]
- code to convert bytes to float16 big endian:
```python
import struct
struct.unpack('>e', bytes([ 0b01001010, 0 ]))
```
returns `12.0`
```
>>> struct.unpack('>e', bytes([0b01001010,0b10000000]))
(13.0,)
>>> struct.unpack('>e', bytes([ 0b11001010, 0 ]))
(-12.0,)
```


## format of the embedding

- type: float 16 bits
- shape: (2, 77, 768)
- constraints: it's twice the same (77, 768) tensor
- prompt_embeds[0, 0, 19] == -28.078125
- prompt_embeds[0, 0, 681] == 33.09375
- other values are (normally?) distributed between approx -12 and 13
- there's no specific indexes that have zero values (0.0 is actually quite rare)

# currently

copy paste all key_to_embedding into the console
create a key:
key = generate_random_base64()
convert to array:
key_bin = convert_key_to_binary(key)
x = convert_bin_key_to_float_array(key_bin)

plot the values:
from matplotlib import pyplot

pyplot.hist(x)
pyplot.show()

check the range is ok

convert this into an image!

## initial image


probably set by `prepare_latents` (in `pipeline_stable_diffusion.py`)

```python
def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)
```

here called:
```python
latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator=None,
            latents=None,
        )
```

notes: vae scale factor is equal to 8

```python
latents = prepare_latents(1, 1, 3, 200, 300, torch.float16, 'cuda', 8)
```

convert latents to image (1st of the batch):
```python
import torchvision
img = torchvision.transforms.functional.to_pil_image(latents[0])
img.show()
```


the actual shape of the latents is: 1, 4, 52, 80
that is 80*52*4 = 16640 numbers
that is 33280 bytes (x2)
that is 499200 encoded bits (x15)
that is 83200 extra characters (/6)

### thing

```python
import torch
import torchvision

from prepare_model import prepare_latents

latents = prepare_latents(1, 1, 3, 400, 600, torch.float16, 'cuda', 8)
img = torchvision.transforms.functional.to_pil_image(latents[0])
img.show()
```

