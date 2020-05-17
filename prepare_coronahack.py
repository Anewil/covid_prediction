from collections import defaultdict
from random import shuffle

import pandas as pd
import shutil
import os

df = pd.read_csv('coronahack/Chest_xray_Corona_Metadata.csv')

groups = defaultdict(list)
groups_covid = defaultdict(list)

for index, row in df.iterrows():
    if row['Label'] == 'Normal':
        item_class = 'normal'
        virus_category = 'healthy'
    else:
        virus_category = 'other'
        if row['Label_1_Virus_category'] == 'bacteria':
            item_class = 'bacteria'
        elif row['Label_1_Virus_category'] == 'Virus':
            item_class = 'virus'
            if row['Label_2_Virus_category'] == 'COVID-19':
                virus_category = 'covid'

        else:
            continue

    if row['Dataset_type'] == 'TRAIN':
        directory = 'train'
    else:
        directory = 'test'

    filename = row['X_ray_image_name']
    old_path = 'coronahack/chest_xray_corona/{}/{}'.format(directory, filename)
    new_path = 'virusorbacteria/{}/{}/{}'.format(directory, item_class, filename)
    groups[(directory, item_class)].append({
        'old_path': old_path,
        'new_path': new_path,
    })
    groups_covid[virus_category].append(old_path)

for item_class in ['normal', 'virus', 'bacteria']:
    data = groups[('train', item_class)]
    shuffle(data)
    percentage_index = int(len(data) * 0.8)
    train = data[:percentage_index]
    val = data[percentage_index:]
    for item in val:
        item['new_path'] = item['new_path'].replace('/train/', '/val/')
    groups[('train', item_class)] = train
    groups[('val', item_class)] = val

for key, value in groups.items():
    for item in value:
        new_path = item['new_path']
        old_path = item['old_path']
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copyfile(old_path, new_path)


def save_seq_to_path(seq, data_type, item_class):
    for old_path in seq:
        new_path = 'covid/{}/{}/'.format(data_type, item_class)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy(old_path, new_path)


for item_class in ['healthy', 'other', 'covid']:
    data = groups_covid[item_class]
    shuffle(data)
    train_index = int(len(data) * 0.6)
    val_index = int(len(data) * 0.8)
    train = data[:train_index]
    save_seq_to_path(train, 'train', item_class)
    val = data[train_index:val_index]
    save_seq_to_path(val, 'val', item_class)
    test = data[val_index:]
    save_seq_to_path(test, 'test', item_class)
