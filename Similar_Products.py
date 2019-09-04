import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
import os
from IPython.display import display, Image
from pylab import imshow, show
from cv2 import imread
import wget
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten
from keras import applications
import sys
from sqlalchemy import Column, String, Integer, create_engine, ForeignKey, BLOB, Date, Boolean, Table, Float, ARRAY, TIMESTAMP, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_mixins import SerializeMixin

Base = declarative_base()

url = "postgresql://jagadeesh:jaggu@127.0.0.1:5432/testing"

def create_tables(url, table_name="similar_products"):
    engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(engine)#, [Base.metadata.tables[table_name]],checkfirst=True)


def download(img_url):
    '''
    Function to Download The Input Image Given in the URL 'img_url'
    '''
    try:
        os.remove('TOP/CLASS/test_img.jpg')
    except:
        pass
    wget.download(img_url, 'TOP/CLASS/test_img.jpg')


def extract_only_image_names(CategoryName_ImageName):
    image_name = []
    for CatName_ImgName in CategoryName_ImageName:
        image_name.append(CatName_ImgName.split('/')[-1])
    return image_name


def get_image_name_of_particular_category(cat_name):
    '''
    Function that returns only the image names of required class and also returns the starting and ending index together in a list stored in the variable 'results'
    '''
    cat_name_check = cat_name.split('/')[-1]
    path = 'FileNames/' + cat_name.split('/')[0] + '/' + cat_name.split('/')[1] + '_file_name.npy'
    CategoryName_ImageName = np.load(path)
    start_index = 0
    for CatName_ImgName in CategoryName_ImageName:
        if str(CatName_ImgName).startswith(cat_name_check):
            break
        start_index += 1
    end_index = start_index
    for i in range(start_index,len(CategoryName_ImageName)):
        if str(CategoryName_ImageName[i]).startswith(cat_name_check):
            end_index += 1
        else:
            break

    CategoryName_ImageName = CategoryName_ImageName[start_index:end_index]
    image_name = extract_only_image_names(list(CategoryName_ImageName))

    return([image_name,start_index,end_index])


def get_image_features(direc):
    datagen = ImageDataGenerator(rescale=1. / 255)
    #Extract Image Features by VGG16
    model = applications.VGG16(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        direc,
        target_size=(224, 224),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train_input = model.predict_generator(generator, 1 // 1)
    bottleneck_features_train_input = bottleneck_features_train_input.reshape(1,25088)
    #Return Features of Input Image
    return bottleneck_features_train_input[0]


def get_similar_image_name_pid_pdists(indices,pdists,image_name):
    index_for_pdists = 0
    similar_image_name = []
    similar_pid = []
    similar_distance = []
    for index in indices:
        print(image_name[index])
        PID = image_name[index].split('_')[0]
        if PID not in similar_pid:
            similar_image_name.append(image_name[index])
            similar_pid.append(PID)
            similar_distance.append(pdists[index_for_pdists])
        index_for_pdists += 1
    return([similar_image_name,similar_pid,similar_distance])


def display_input_image():
    img_loc = 'TOP/CLASS/test_img.jpg'
    imshow(imread(img_loc))
    show()


def store_in_db(url,pid,similar_pids):
    engine = create_engine(url, pool_pre_ping=True)
    similar_pdt = ",".join(similar_pids)
    query = " INSERT INTO public.similar_products VALUES(" + "'{" + pid + "}'" + ',' + "'{" + similar_pdt + "}') "
    engine.execute(query)


def show_table():
    engine = create_engine(url, pool_pre_ping=True)
    query = "SELECT * FROM public.similar_products"
    rows = engine.execute(query)
    print("PID\t\tSimilar_PID")
    for row in rows:
        print(row[0],'\t\t',row[1])

# def display_similar_products(pid,cat_name,similar_image_name, similar_PID, similar_distance):
#     cursor = engine.connect()
#     for i in range(len(similar_image_name)):
#         query = '''SELECT pid,url,brand,product_title FROM products where pid = '{}' '''.format(similar_PID[i])
#         pd_list = cursor.execute(query).fetchall()
#         rows = pd.DataFrame(pd_list,columns = ['pid','url','brand','product_title'])
#         title =list(rows['product_title'])
#         url = list(rows['url'])
#         pid = list(rows['pid'])
#         brand = list(rows['brand'])
#         img_loc = cat_name + '/' + similar_image_name[i] +'.jpg'
#         #print(img_loc)
#         imshow(imread('/home/ec2-user/SageMaker/level_2_decoded/'+img_loc))
#         show()
#         print("Product Title : ",title[0])
#         print("Product URL : ",url[0])
#         print("Product ID : ",pid[0])
#         print("Brand : ",brand[0])
#         print("Euclidean Distance : ",similar_distance[i])


def get_similar_products_cnn(bottleneck_features_train,pid, cat_name, img_url, num_results):
    download(img_url) #Download the Input Image
    image_name, start_index, end_index = get_image_name_of_particular_category(cat_name) #Get only the particular category's image name
    input_img_feature = get_image_features('TOP') #detect the Features of the Input Image downloaded in the directory 'TOP'

    bottleneck_features_train_given_category = bottleneck_features_train[start_index:end_index]#Loading the Features of trained images in a numpy file
    pairwise_dist = pairwise_distances(bottleneck_features_train_given_category, input_img_feature.reshape(1,-1))#Finding Euclidean Distance Between the Input Image and Trained Images

    indices = np.argsort(pairwise_dist.flatten())[0:num_results] #Storting and storing the indices based on distance
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results] #Storting and storing the distances

    similar_image_name, similar_pid, similar_distance = get_similar_image_name_pid_pdists(indices,pdists,image_name)
    # print(similar_pid)
    # print("INPUT IMAGE\nCATEGORY : ",cat_name)
    # display_input_image() #Function call to Display Input Image
    # display_similar_products(pid,cat_name,similar_image_name, similar_PID, similar_distance) #Function call to display Similar Products
    store_in_db(pid,similar_pid) #Store the Similar Product's PID in DB


def generate_similar_products(pid,img_url,hierarchy):
    cat_name = hierarchy
    feature_file_path = 'Image_Features/' + cat_name.split("/")[0] + '/' + cat_name.split("/")[1] + '.npy'
    bottleneck_features_train = np.load(feature_file_path)
    num_results = 10
    get_similar_products_cnn(bottleneck_features_train,pid,cat_name,img_url,num_results)


def main():
    pid = sys.argv[0]
    image_url = sys.argv[1]
    hierarchy = sys.argv[2]
    #generate_similar_products(pid,image_url,hierarchy)
    create_tables(url,"similar_products")
    store_in_db(url,"1",["1","2","3"])

if __name__ == "__main__":
    main()
