import json
import os

import numpy as np

from PIL import Image


def get_dataset(path):
    for image_path in os.listdir(path):
        print(image_path)

def get_label_info(label_template_info):
    with open(label_template_info) as json_file:
        return json.load(json_file)

def generate_six_class_index(label_template_info):

    excluded_labels = []

    included_classes = [
        'Belt',
        'Filet',
        'Black lining',
        'Nematodes',
        'Blood',
        'Level of bleeding (overall)'
    ]

    six_class_index = {}

    label_info = get_label_info(label_template_info)
    
    idx = 0

    for _, top_key in enumerate(label_info):
        labels = []
        print(top_key)
        for low_key in label_info[top_key]:
            if low_key != 'color':
                labels.append(label_info[top_key][low_key])

        if top_key in included_classes:
            six_class_index[top_key] = {
                'labels': labels,
                'label': idx
            }

            idx += 1
        else:
            excluded_labels.extend(labels)
    
    six_class_index['Filet']['labels'].extend(excluded_labels)

    return six_class_index

def reverse_index(index):
    new_index = {}
    for key in index:
        for value in index[key]['labels']:
            new_index[value] = index[key]['label']
    return new_index

def gen_masks_to_six_classes():
    index = generate_six_class_index('./datasets/marel/RGBdata/labelTemplate.json')

    reversed_index = reverse_index(index)

    for image_path in os.listdir('./datasets/marel/RGBdata/mask'):
        mask = np.array(Image.open('./datasets/marel/RGBdata/mask/'+image_path))
        new_mask = np.vectorize(reversed_index.get)(mask)
        
        Image.fromarray(new_mask.astype(np.uint8)).save('./datasets/marel/RGBdata/six_class_mask/'+image_path)


def splitPerc(l, perc):
        # Turn percentages into values between 0 and 1
        splits = np.cumsum(perc)/100.

        if splits[-1] != 1:
            raise ValueError("percents don't add up to 100")

        # Split doesn't need last percent, it will just take what is left
        splits = splits[:-1]

        # Turn values into indices
        splits *= len(l)

        # Turn double indices into integers.
        # CAUTION: numpy rounds to closest EVEN number when a number is halfway
        # between two integers. So 0.5 will become 0 and 1.5 will become 2!
        # If you want to round up in all those cases, do
        # splits += 0.5 instead of round() before casting to int
        splits = splits.round().astype(np.int)

        train = l[0:splits[0]]
        val = l[splits[0]:splits[1]]
        test = l[splits[1]:]

        return train, val, test

def save_txt(paths, mode):
    results = []

    for path in paths:
        results.append(f'RGBdata/org/{path},RGBdata/six_class_mask/{path}')

    results = np.array(results)

    np.savetxt(f'./datasets/marel/{mode}.txt', results, fmt="%s")

def generate_train_val_test_txt(path, percentages):

    paths = np.array(os.listdir(path))

    np.random.shuffle(paths)

    train, val, test = splitPerc(paths, percentages)

    save_txt(train, 'train')
    save_txt(val, 'val')
    save_txt(test, 'test')



if __name__ == "__main__":

    path = './../datasets/marel/RGBdata/org'
    percentages = [80, 5, 15]

    generate_train_val_test_txt(path, percentages)
