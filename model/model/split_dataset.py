#
# Created on Sun Nov 13 2022
#
# python3 split_dataset.py --data_path=/path/to/dataset --new_dataset_name=Name_of_dataset --size=No_of_images_per_class --train_ratio=split_ratio --shuffle=True/false
#


import argparse
import os
import json
import random
import numpy as np
import shutil

def data_json_file(data):
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def main(data_path: str, new_dataset_name:str,size:int, split_ratio: float, shuffle=False):
    rootdir = data_path
    dataset_name =  new_dataset_name
    sub_dir = os.listdir(rootdir)
    sub_dirs_path = [os.path.join(rootdir, x) for x in sub_dir]
    print("Generating dataset info")
    # Generating dataset info
    dataset_info_by_class = []
    for path in sub_dirs_path:
        data_class = path.split('/')[-1]
        images = list(filter(lambda x: x.endswith('.jpg'), os.listdir(path)))
        dataset_info_by_class.append({"class": data_class, "total_images": len(images), "images": images})
    data_json_file(dataset_info_by_class)
    # Genration of new dataset 
    if size:
        N = size 
        total_classes = [dataset_info_by_class[i]["total_images"] for i in range(10)]
        min_possible_N = min(total_classes)
        max_possible_N = max(total_classes)
        print(min_possible_N, max_possible_N)
        if N not in range(min_possible_N, max_possible_N+1):
            raise ValueError("Minimum possible value for N is {min_n} and Maximum possible value for N is {max_n}".format(min_n = min_possible_N, max_n=max_possible_N ))
        
    root = "/".join(rootdir.split('/')[:-1])
    os.makedirs( os.path.join(root,dataset_name),  exist_ok=True)
    os.makedirs(os.path.join(root,dataset_name,"Train"), exist_ok=True)
    os.makedirs(os.path.join(root,dataset_name,"Test"), exist_ok=True)
    for data in dataset_info_by_class:
        total_images = list(set(data["images"]))
        if shuffle:
            random.shuffle(total_images)
        images = total_images[:size] if size else total_images
        if shuffle:
            random.shuffle(images)
        #test_counter = np.round(data["total_images"] * (1 - split_ratio))
        test_counter = np.round(size * (1 - split_ratio)) if size else np.round(data["total_images"] * (1 - split_ratio))
        split= int(len(images)-test_counter)
        train_images = images[:split]
        test_images = images[split:]
        os.makedirs(os.path.join( root, dataset_name,"Train",data["class"]), exist_ok=True)
        os.makedirs(os.path.join( root, dataset_name,"Test",data["class"]), exist_ok=True)
        for image in train_images:
            src_jpg = os.path.join(rootdir,data["class"],image)
            dst_path = os.path.join(root,dataset_name,"Train",data["class"])
            shutil.copy(src_jpg, dst_path)
        for image in test_images:
            src_jpg = os.path.join(rootdir,data["class"],image)
            dst_path = os.path.join(root,dataset_name,"Test",data["class"])
            shutil.copy(src_jpg, dst_path)
        
        print(len(train_images),  len(test_images))
    pass

def parse_args():
    parser = argparse.ArgumentParser(description = "Dataset divider")
    parser.add_argument("--data_path", required = True, help = "Path to data")
    parser.add_argument("--new_dataset_name", required = True, help = "Name of new dataset")
    parser.add_argument("--size", help = "Required images per class")
    parser.add_argument("--train_ratio", required = True, help = "Train ration - 0.7 means 70-30 split between testing and training data")
    parser.add_argument("--shuffle", help = "Shuffle dataset")
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    split_ratio = args.train_ratio
    shuffle = args.shuffle
    new_dataset_name = args.new_dataset_name
    size = args.size if args.size else 0
    main(data_path, new_dataset_name, int(size), float(split_ratio), shuffle)
    