import argparse
import gc
import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from diffusers import StableDiffusionPipeline
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm

sys.path.append('.')
from prepare_model import prepare_config, prepare_model


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--plot-pca', action='store_true')
arg_parser.add_argument('--random-embeddings-file', type=str, default='random-embeddings.npy')
arg_parser.add_argument('--nb-random-embeddings', type=int, default=56)
arg_parser.add_argument('--texts-file', type=str, default='ignored/texts1.txt')
arg_parser.add_argument('--pca-outliers-idx-file', type=str, default='ignored/pca-outliers.txt')
parsed_args = arg_parser.parse_args()

do_plot_embeddings = parsed_args.plot_pca
texts_file_path = parsed_args.texts_file
# note: you need to do the pca 1st to know which are outliers
pca_outliers_indexes_file_path = parsed_args.pca_outliers_idx_file
nb_random_embeddings = parsed_args.nb_random_embeddings
random_embeddings_file_path = parsed_args.random_embeddings_file

def embed_prompt(pipe: StableDiffusionPipeline,
                  prompt,
                  device,
                  num_images_per_prompt,
                  do_classifier_free_guidance) -> torch.Tensor:
    with torch.no_grad():
        prompt_embeds = pipe._encode_prompt(prompt,
                                            device,
                                            num_images_per_prompt,
                                            do_classifier_free_guidance,
                                            None)
    return prompt_embeds

class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


config = prepare_config()
model_name = config['model_name']
device = config['device']
num_inference_steps = config['num_inference_steps']
prompt = config['prompt']
height = config['height']
width = config['width']
guidance_scale = config['guidance_scale']
output_type = config['output_type']
batch_size = config['batch_size']
num_images_per_prompt = config['num_images_per_prompt']
dtype = config['dtype']
do_classifier_free_guidance = config['do_classifier_free_guidance']
pipe = prepare_model(model_name, dtype, device)



#embeds_batch_size = 1024
#embeds_batch_size = 12
with open(texts_file_path, 'r') as f:
    texts = f.read().strip().split('\n')

texts_dataset = TextDataset(texts)
#texts_batches = DataLoader(texts_dataset, batch_size=embeds_batch_size, shuffle=False)

all_the_embeddings = []
batch_ct = 0
for prompt in tqdm.tqdm(texts_dataset):
    embedding = embed_prompt(pipe, prompt, device, num_images_per_prompt, do_classifier_free_guidance)
    all_the_embeddings.append(embedding.flatten().cpu().numpy())
    batch_ct += 1
    #if(batch_ct > 20):
    #    break

#print(f'stacking {len(all_the_embeddings)} batches of size {embeds_batch_size}')
print(f'stacking {len(all_the_embeddings)} embeddings')
embeddings = np.stack(all_the_embeddings)

if(pca_outliers_indexes_file_path != ''):
    with open(pca_outliers_indexes_file_path, 'r') as pca_outliers_indexes_file:
        pca_outliers_indexes_file_content = pca_outliers_indexes_file.read()
    pca_outliers_indexes = [
        int(idx_txt.strip())
        for idx_txt in pca_outliers_indexes_file_content.split(',')
    ]
    embeddings = np.delete(embeddings, pca_outliers_indexes, axis=0)

print('freeing some memory')
all_the_embeddings = None
texts = None
texts_dataset = None
pipe = None
time.sleep(2.2)
cleaned_ct = gc.collect()
print(f'collect returned: {cleaned_ct}')


if(do_plot_embeddings):
    pca = PCA(n_components=3)
    print(f'fitting PCA')
    pca.fit(embeddings)
    #plot_sample_size = 1102
    #sampling_indexes = np.random.randint(low=0,  high=len(embeddings)-1, size=plot_sample_size)
    #plot_sample = embeddings[sampling_indexes]
    plot_sample = embeddings
    print(f'transforming {len(plot_sample)} vectors')
    projected_emebddings = pca.transform(plot_sample)
    print('plotting')
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(projected_emebddings[:,0], projected_emebddings[:,1], projected_emebddings[:,2])
    fig.show()
    projected_emebddings = None
    pca = None
    time.sleep(1.8)
    gc.collect()

def create_random_embeddings(models: np.ndarray, n: int):
    nb_weights = len(models)
    weights = np.random.sample((n, nb_weights))
    weights[:,] /= weights.sum(1)[:,np.newaxis]
    random_embeddings = np.matmul(weights, models)
    return random_embeddings

if((nb_random_embeddings > 0) and (random_embeddings_file_path != '')):
    print(f'creating {nb_random_embeddings} random embeddings')
    random_embeddings = create_random_embeddings(embeddings, nb_random_embeddings)
    print(f'saving random embeddings to {random_embeddings_file_path}')
    np.save(random_embeddings_file_path, random_embeddings, allow_pickle=False)

print('all done')