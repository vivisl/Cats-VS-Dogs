# Cats-VS-Dogs
Classification for cats and dogs with Tensorflow implementation
# Outline
* [Data organization](#Data-organization)
* [Data preprocessing](#Data preprocessing)
* [Building the classifier](#Building the classifier)
* [Create the Estimator](#contact)
* [Training and Evaluating](#Training and Evaluating)

# Data organization
1. create a data folder in which we'll have in 2 subfolders: ./Train and ./Validation. Each of them will have 2 folders: ./cats and ./Dogs. This folder structure allows the model to know where to fetch the images and their labels for either training or validation folder.

![alt tag](http://a3.qpic.cn/psb?/V13jsLBD3Y4Bf1/16upB2c3Rz2RsWqLVxGjWktBuG.UDCbG*DWR1kLl.vk!/b/dLYAAAAAAAAA&ek=1&kp=1&pt=0&bo=IwFMAgAAAAARF0w!&tl=3&vuin=442394235&tm=1566381600&sce=60-1-1&rf=viewer_4)

```
def organize_datasets(src_train_path, dest_train_path, n= 25000, ratio=0.2):
    src_path = pathlib.Path(src_train_path)
    # get all jpg files in this src_path path 
    files = list(src_path.glob('*.jpg'))
    random.shuffle(files)    
    files = files[:n]
    
    n = int(len(files) * ratio)
    val, train = files[:n], files[n:]
   
    # remove old training folders
    shutil.rmtree(dest_train_path, True) 
    print('{} removed'.format(dest_train_path))
    
    # create new training folders
    for c in ['dogs', 'cats']: 
        os.makedirs('{}/train/{}/'.format(dest_train_path, c))
        os.makedirs('{}/validation/{}/'.format(dest_train_path, c))
    print('folders created !')

    # determine the picture's name is cat or dog
    for t in train:
        if 'cat' in t.name:
            shutil.copy2(t, os.path.join(dest_train_path, 'train', 'cats'))
        else:
            shutil.copy2(t, os.path.join(dest_train_path, 'train', 'dogs'))
                
    for v in val:
        if 'cat' in v.name:
            shutil.copy2(v, os.path.join(dest_train_path, 'validation', 'cats'))
        else:
            shutil.copy2(v, os.path.join(dest_train_path, 'validation', 'dogs'))
    print('Data copied!')    
    
src_train = './dog_cat/train'
training_path = './dog_cat_data'
organize_datasets(src_train, training_path)
```
2. Fetch the images and their labels
```
def get_files(dogs, cats):        
    dogs_path = pathlib.Path(dogs)
    cats_path = pathlib.Path(cats)
    
    dogs_list = list(dogs_path.glob('*.jpg'))
    cats_list = list(cats_path.glob('*.jpg'))
    file_list_add = dogs_list + cats_list
    random.shuffle(file_list_add)
    file_list = [str(i) for i in file_list_add]  
    
    label_list = []
    for t in file_list_add:
        if 'cat' in t.name:
            label_list.append(0) 
        else:
            label_list.append(1) 
    
    return file_list, label_list

dogs_train = './dog_cat_data/train/dogs'
cats_train = './dog_cat_data/train/cats'
train_file_list, train_label_list = get_files(dogs_train, cats_train)

dogs_validation = './dog_cat_data/validation/dogs'
cats_validation = './dog_cat_data/validation/cats'
validation_file_list, validation_label_list = get_files(dogs_validation, cats_validation)
```
