# Cats-VS-Dogs
Classification for cats and dogs with Tensorflow implementation
# Outline
* [Data organization](#Data-organization)
* [Data preprocessing](#Data-preprocessing)
* [Epochs processing](#Epochs-processing)
* [Building the classifier](#Building-the-classifier)
* [Create the Estimator](#Create-the-Estimator)
* [Training and Evaluating](#Training-and-Evaluating)

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
# Data preprocessing
When training a neural network on real-world image data, it is often necessary to convert images of different sizes to a common size, so that they may be batched into a fixed size.
Reads an image from a file, decodes it into a dense tensor, and resizes it to a fixed shape.
```
def _parse_function(filename, label):
    image_H = image_W = 150
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)  
    image_resized = tf.image.resize_images(image_decoded, [image_H, image_W])
    img_std = tf.image.per_image_standardization(image_resized)
    return img_std, label
```

Preprocess data with Dataset.map( )
The Dataset.map(f) transformation produces a new dataset by applying a given function f to each element of the input dataset.
```
    filenames = tf.constant(validation_file_list)
    validation_lables = tf.constant(validation_label_list)
   
    dataset = tf.data.Dataset.from_tensor_slices((filenames , validation_lables))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=1000)
```
# Epochs processing
the simplest way to iterate over a dataset in multiple epochs is to use the Dataset.repeat() transformation. 
batch_size = 50
```
def train_input_fn():
    '''
    運行dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))後，dataset的一個元素是(filename, label)。
    filename是圖片的文件名，label是圖片對應的標籤。
    
    之後通過map，將filename對應的圖片讀入，並縮放為150x150的大小。此時dataset中的一個元素是(image_resized, label)。    
    dataset.shuffle(buffersize=1000).batch(32).repeat()的功能是：在每個epoch內將圖片打亂組成大小為32的batch，無限重複。
    
    最後，dataset中的一個元素是(image_std_batch, label_batch)
    image_std_batch(32, 150, 150, 3)，而label_batch的形狀為(32, )，接下來我們就可以用這兩個Tensor來建立模型了。
    
    '''
    global train_file_list
    global train_label_list
    global num_epochs
    global batch_size    
    
    filenames = tf.constant(train_file_list)
    train_lables = tf.constant(train_label_list)   
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames , train_lables))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
        
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    
    return features, labels
```

Validate all the images in the validation folder once In validation_input_fn
batch_size = 50
```
def validation_input_fn():
    global validation_file_list
    global validation_label_list
    global num_epochs
    global batch_size
    
    filenames = tf.constant(validation_file_list)
    validation_lables = tf.constant(validation_label_list)
   
    dataset = tf.data.Dataset.from_tensor_slices((filenames , validation_lables))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=1000)
    
    #dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
 
    iterator = dataset.make_one_shot_iterator()
    validation, v_lables = iterator.get_next() 
    
    return validation, v_lables
```
# Building the classifier
1. Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5‐pixel subregions), with ReLU activation function
2. Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
3. Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
4. Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
5. Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4
6. Dense Layer #2 (Logits Layer): 2 neurons, one for each digit target class (0 or 1).

```
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 150, 150, 3])
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
       
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 37 * 37 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }    
    print('predictions:', predictions)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        starter_learning_rate = 0.001
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   decay_steps=500,
                                                   decay_rate=0.9,
                                                   staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
            
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)  
```
# Create the Estimator
The model_fn argument specifies the model function to use for training, evaluation, and prediction.

```
# Create the Estimator
dog_cat_classifier = tf.estimator.Estimator( model_fn=cnn_model_fn, model_dir="./cnn_model")
```
# Training and Evaluating
20,000 images in train folder and 5,000 images in validation folder
batch_size = 50
3000 train steps
Validate all the validation images once

```
# Train the model
#train_result = dog_cat_classifier.train( input_fn=train_input_fn, steps=3000, hooks=[logging_hook])
train_result = dog_cat_classifier.train( input_fn=train_input_fn, steps=3000)
print('train_result:', train_result)

# Evaluate the model and print results
#eval_results = dog_cat_classifier.evaluate(input_fn=validation_input_fn, steps=100)
eval_results = dog_cat_classifier.evaluate(input_fn=validation_input_fn)
print('eval_results:', eval_results)  
print('----------ALL Done!----------')
```
