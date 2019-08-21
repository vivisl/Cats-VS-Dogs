# Cats-VS-Dogs
Classification for cats and dogs with Tensorflow implementation
# Outline
* [Data organization](#Data-organization)
* [Data preprocessing](#Data preprocessing)
* [Building the classifier](#Building the classifier)
* [Create the Estimator](#contact)
* [Training and Evaluating](#Training and Evaluating)

# Data organization
create a data folder in which we'll have in 2 subfolders: ./Train and ./Validation

Each of them will have 2 folders: ./cats and ./Dogs

This structure will allow our model to know from which folder to fetch the images as well as their labels for either training or validation.

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
