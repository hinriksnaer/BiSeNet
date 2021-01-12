import os
import csv
import pandas as pd

def main(dataset_path, dataset_mode):

    data_label_pairs = []

    data_path = 'leftImg8bit/'
    label_path = 'gtFine/'
    
    for subdirectory in os.listdir(dataset_path + '/' + data_path + dataset_mode):
        for image_path in os.listdir(dataset_path + '/' + data_path + dataset_mode + '/' + subdirectory):
            image_label_path = image_path.split('_')[0:-1]
            image_label_path.append('gtFine_labelIds.png')
            image_label_path = '_'.join(image_label_path)

            data_label_pairs.append(data_path +  dataset_mode + '/' + subdirectory + '/' + image_path+','+ label_path + dataset_mode + '/' + subdirectory + '/' + image_label_path)

    df = pd.DataFrame(data_label_pairs)
    df.to_csv('./datasets/cityscapes' + '/' + dataset_mode + '.txt', index=False, header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar=' ')

if __name__ == "__main__":
    dataset_path = './datasets/cityscapes'
    dataset_mode = 'test'
    
    main(dataset_path, dataset_mode)