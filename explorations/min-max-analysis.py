



from matplotlib import pyplot
import torch
import tqdm

from prepare_model import prepare_config, prepare_model


def embed_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance):
    with torch.no_grad():
        prompt_embeds = pipe._encode_prompt(prompt,
                                            device,
                                            num_images_per_prompt,
                                            do_classifier_free_guidance,
                                            None)
    return prompt_embeds


config = prepare_config()
model_name = config['model_name']
device = config['device']
num_inference_steps = config['num_inference_steps']
prompt = config['prompt']
height = config['height']
width = config['width']
guidance_scale = config['guidance_scale']
batch_size = config['batch_size']
num_images_per_prompt = config['num_image_per_prompt']
dtype = config['dtype']
do_classifier_free_guidance = config['do_classifier_free_guidance']
pipe = prepare_model(model_name, dtype, device)


with open('ignored/texts1.txt', 'r') as f:
    texts = f.read().strip()

texts = texts.split('\n')


def get_where_special_values(x):
    masked = torch.where((x<-28)|(x>33), x, 0.0)
    #masked = torch.where(masked < 0., -1, masked)
    #masked = torch.where(masked > 0., 1., masked)
    return (torch.argwhere(masked < 0).tolist(), torch.argwhere(masked > 0).tolist())

def get_zero_indexes(x: torch.Tensor) -> torch.Tensor:
    return torch.argwhere(x == 0.0).detach().cpu()

embed_maxs = []
embed_mins = []
where_specials = []
where_zeros = []


with torch.no_grad():
    for prompt in tqdm.tqdm(texts):
        x = embed_prompt(prompt, device, 1, False)
        where_specials.append(get_where_special_values(x))
        #x = torch.where(x > -25, x, 0.0)
        #x = torch.where(x < 30, x, 0.0)
        zero_indexes = get_zero_indexes(x)
        if(zero_indexes.shape[0] > 0):
            where_zeros.append(zero_indexes)
        x = torch.where(x > -28.078124, x, 0.0)
        x = torch.where(x < 33.09374, x, 0.0)
        embed_maxs.append(x.max().item())
        embed_mins.append(x.min().item())


print(f'min={min(embed_mins)}')
print(f'max={max(embed_maxs)}')

pyplot.hist(embed_maxs)
pyplot.title('maxes')
pyplot.show()

pyplot.hist(embed_mins)
pyplot.title('mines')
pyplot.show()

special_min_indexes, special_max_indexes = zip(*where_specials)
all([t == [[0,0,19]] for t in special_min_indexes])
# -> True
all([t == [[0,0,681]] for t in special_max_indexes])
# -> True

print(where_zeros)

# special values min=-28.078125 - max=33.09375
# without the special values: min=-11.953125, max=12.703125