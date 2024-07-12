import math
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scipy.io
import pickle


def read_column(csv_file_path,col_num):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    # Get the first column's name. This assumes the CSV has a header row.
    first_column_name = df.columns[col_num]
    # If you know the name of the first column, you can skip the above step and use the name directly.
    # Select the first column using its name
    first_column_data = df[first_column_name]
    # If you want the data as a list:
    first_column_list = first_column_data.tolist()
    # Print the data from the first column
    return first_column_list

def split_substring(substring,desc_list):
    # Find indices of elements containing 'img1'
    indices = [index for index, element in enumerate(desc_list) if substring in element]
    return indices

def convert_to_few_shot(args,description,images_path):
    images_path = [path[1:] for path in images_path]
    indices_description_shots = split_substring("img1",images_path)
    indices_description_test = split_substring("img2",images_path)
    description_shots = [description[i] for i in indices_description_shots]
    description_test = [description[i] for i in indices_description_test]
    images_path_shots = [args.image_folder+images_path[i] for i in indices_description_shots]
    images_path_test = [args.image_folder+images_path[i] for i in indices_description_test]
    return description_shots,description_test,images_path_shots,images_path_test

class FSDataset(Dataset):
    def __init__(self,args,image_processor):
        """
        Args:
            data_list (list): List of your data.
        """
        with open(args.episodes_path, 'rb') as f:
            episodes = pickle.load(f)
        # Initialize lists to hold results
        if args.chunks > 1:
            episodes = episodes[args.curr_chunk:-1:args.chunks]

        all_image_paths = []
        all_class_names = []

        # Process each item in the list
        for data in episodes:
            # Initialize temporary lists for current item
            image_paths = []
            class_names = []

            # Add positive images and class name
            image_paths.extend(data['positive_images'])
            class_names.extend([data['test_class']])

            # Add negative images and class names
            for neg in data['negs']:
                image_paths.extend(neg['neg_images'])
                class_names.extend([neg['neg_class']])

            # Add the test image and class at the end
            image_paths.extend([data['test_image']])
            class_names.extend([data['test_class']])

            # Append results to the overall lists
            all_image_paths.append(image_paths)
            all_class_names.append(class_names)

        self.images_path = all_image_paths
        self.description = all_class_names
        self.args = args
        self.image_processor = image_processor

    def __len__(self):
        return self.images_path.__len__()

    def __getitem__(self, idx):# every image and desc used once. same order everytime for now
        image_shot_tensor_list =[]
        image_test_tensor_list =[]
        descriptions = self.description[idx]
        images = self.images_path[idx]
        desc_shots_list = descriptions[0:2]
        desc_test_list = [descriptions[2]]
        for i in range(0,2):
            image_shot = Image.open(images[i])
            curr_image_shot = self.image_processor.preprocess(image_shot, return_tensors='pt')['pixel_values'][0]
            image_shot_tensor_list.append(curr_image_shot)
        image_test = Image.open(images[2])
        curr_image_test = self.image_processor.preprocess(image_test, return_tensors='pt')['pixel_values'][0]
        image_test_tensor_list.append(curr_image_test)

        images_shot_tensors = torch.stack(image_shot_tensor_list, dim=0)
        images_test_tensors = torch.stack(image_test_tensor_list, dim=0)

        return images_shot_tensors,desc_shots_list,images_test_tensors,desc_test_list,images

def get_dataloader(args,image_processor):
    fs_dataset = FSDataset(args,image_processor)
    fs_dataloader = DataLoader(fs_dataset, batch_size=args.bs, shuffle=False)
    return fs_dataloader
