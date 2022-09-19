import os, shutil
from  keras import layers
from keras import models
from keras_preprocessing.image import ImageDataGenerator

original_dataset_dir = "C:/Users/patricia/Desktop/vehiclepark/data"
base_dir = "C:/Users/patricia/Desktop/vehiclepark/data/free_and full_availables"
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_free_dir = os.path.join(train_dir, 'free')
os.mkdir(train_free_dir)

train_full_dir = os.path.join(train_dir, 'full')
os.mkdir(train_full_dir)

validation_free_dir = os.path.join(validation_dir, 'free')
os.mkdir(validation_free_dir)

validation_full_dir = os.path.join(validation_dir, 'full')
os.mkdir(validation_full_dir)

test_free_dir = os.path.join(test_dir, 'free')
os.mkdir(test_free_dir)

test_full_dir = os.path.join(test_dir, 'full')
os.mkdir(test_full_dir)

# fnames = ['free.{}.jpg'.format(i) for i in range(100)]
# for fnames in fnames:
#     src = os.path.join(original_dataset_dir, fnames)
#     dst = os.path.join(train_free_dir,fnames)
#     shutil.copyfile(src,dst)
#
# fnames = ['free.{}.jpg'.format(i) for i in range(100,150)]
# for fnames in fnames:
#     src = os.path.join(original_dataset_dir,fnames)
#     dst = os.path.join(validation_free_dir,fnames)
#     shutil.copyfile(src,dst)
#
# fnames = ['free.{}.jpg'.format(i) for i in range(150,200)]
# # for fnames in fnames:
#     src = os.path.join(original_dataset_dir, fnames)
#     dst = os.path.join(test_free_dir,fnames)
#     shutil.copyfile(src,dst)
#
# fnames = ['full.{}.jpg'.format(i) for i in range(100)]
# for fnames in fnames:
#     src = os.path.join(original_dataset_dir,fnames)
#     dst = os.path.join(train_full_dir,fnames)
#     shutil.copyfile(src,dst)
# #
# fnames = ['full.{}.jpg'.format(i) for i in range(100,150)]
# for fnames in fnames:
#     src = os.path.join(original_dataset_dir,fnames)
#     dst = os.path.join(validation_full_dir,fnames)
#     shutil.copyfile(src,dst)
#
# fnames = ['full.{}.jpg'.format(i) for i in range(150,200)]
# for fnames in fnames:
#     src = os.path.join(original_dataset_dir,fnames)
#     dst = os.path.join(test_full_dir, fnames)
#     shutil.copyfile(src,dst)

# print('total training full images:',len(os.listdir(train_full_dir)))

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='softmax'))

train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
validation_generator=test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
# history=model.fit_generator(
#     train_generator,
#     steps_per_epoch=10,
#     epochs=30,
#     validation_data=validation_generator,
#     validation_steps=50
# )
