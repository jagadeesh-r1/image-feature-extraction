import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
import os
import gc
gc.collect()

from keras.preprocessing.image import ImageDataGenerator
from keras import applications

def save_image_features(category,sub_category):
    print(category,sub_category)
    start_time = time.time()
    img_width, img_height = 224, 224
    train_data_dir = '/home/ubuntu/level_3/'+category + '/' + sub_category
    batch_size = 1
    print(train_data_dir)
    #Function to compute VGG-16 CNN for image feature extraction.

    #To store Class_Name along with Image_Name
    CategoryName_and_ImageName = []

    datagen = ImageDataGenerator(rescale=1. / 255)

    model = applications.VGG16(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        train_data_dir + '/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    for i in generator.filenames:
        print(i)
        CategoryName_and_ImageName.append(i[0:-4])
    #img_feature_file_path = '/home/ubuntu/FileNames/' + category + '/' + sub_category + '_file_name.npy'
    img_feature_file_path = '/home/ubuntu/Image_Features/' + category + '/' + sub_category + '.npy'
    file = open(img_feature_file_path, 'wb+')
    file.close()
    print('file created!')
    #bottleneck_features_train = model.predict_generator(generator, len(generator.filenames) // batch_size)
    #bottleneck_features_train = bottleneck_features_train.reshape((len(generator.filenames),25088))

    #print(bottleneck_features_train)

    np.save(open(img_feature_file_path, 'wb+'), bottleneck_features_train) #Save Image Features in numpy file

    #category_name_path = 'FileNames/' + category + '/' + sub_category + '_file_name.npy'
    #np.save(open(category_name_path, 'wb'), np.array(CategoryName_and_ImageName))
    print('saved numpy file')
    end_time = time.time()

    print('Total Time taken to extract features ===>>  ',end_time-start_time,"\n")


if __name__ == "__main__":
    list = []
    images_path = 'level_3/'
    for i in os.listdir(images_path):
        try:
            img_feature_dir_path = '/home/ubuntu/Image_Features/' + i
            os.mkdir(img_feature_dir_path)
        except:
            print(i,' folder coulnt be created!')
        try:
            for j in os.listdir( images_path +i+'/'):
                if j.startswith('.') or i.startswith('.'):
                    continue
                else:
                    list.append([i,j])
        except:
            pass
    gc.collect()
    for i in range(1,len(list)):
        print(i,list[i])
    list.reverse()
    missed_categories = []

    for i in range(len(list)):
        time.sleep(3)
        try:
            list[i][0] = list[i][0].replace(" ","_")
            list[i][1] = list[i][1].replace(" ","_")
            list[i][0] = list[i][0].replace("&","and")
            list[i][1] = list[i][1].replace("&","and")        
            #print(list[i][0],list[i][1])
            save_image_features(list[i][0],list[i][1])
            gc.collect()
        except:
            missed_categories.append(i)





