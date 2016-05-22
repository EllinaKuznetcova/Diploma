# Diploma

To start working with project you need to install git large files storage extension if you don't have it:

    brew install git-lfs

1. Clone project
2. Enter cloned folder
3. git lfs pull

To start comparing image with processed images from database:

  Parameters:
  
        `result_images_count` - number of images to receive as result.
        
        `enable_query_expansion` - should enable query expansion
        
        `query_image_path` - path to query image

To start process images from database and creating inverted file:

    python videoGoogle.py

