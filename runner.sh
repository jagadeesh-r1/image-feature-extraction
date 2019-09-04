pip3 install keras tensorflow wget --user

aws s3 sync 's3://new-crawling-data/Amazon USA/images/level_3' $level_3_dir

python3 Feature_Detector.py

aws s3 cp Image_Features/ 's3://crawled-images-rec/Image_Features'
aws s3 cp FileNames/ 's3://crawled-images-rec/FileNames'

python3 Similar_Products.py pid image_url hierarchy #the args should be taken from other program.