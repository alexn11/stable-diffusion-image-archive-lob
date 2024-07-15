
# copy paste here everything from base-run.py up to the with torch no grad bit & avoid to paste the argpasse bit as well
(...)
#


from matplotlib import pyplot
import torch


with torch.no_grad():
    pipe = load_model(model_name, dtype, device)

with open('ignored/texts1.txt', 'r') as f:
    texts = f.read().strip()

texts = texts.split('\n')


def get_where_special_values(x):
    masked = torch.where((x<-28)|(x>33), x, 0.0)
    #masked = torch.where(masked < 0., -1, masked)
    #masked = torch.where(masked > 0., 1., masked)
    return (torch.argwhere(masked < 0).tolist(), torch.argwhere(masked > 0).tolist())


embed_maxs = []
embed_mins = []
where_specials = []


with torch.no_grad():
    for prompt in texts:
        x = embed_prompt(prompt, device, 1, False)
        where_specials.append(get_where_special_values(x))
        #x = torch.where(x > -25, x, 0.0)
        #x = torch.where(x < 30, x, 0.0)
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


# special values min=-28.078125 - max=33.09375
# without the special values: min=-11.953125, max=12.703125