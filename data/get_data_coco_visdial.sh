#!/bin/bash

# get VisDial training set
if [ ! -f visdial_0.9_train.json ]
then
    wget https://s3.amazonaws.com/visual-dialog/v0.9/visdial_0.9_train.zip
    unzip visdial_0.9_train.zip
    rm visdial_0.9_train.zip
    mkdir -p visdial
    mv visdial_0.9_train.json visdial/
fi

# get VisDial validation set
if [ ! -f visdial_0.9_val.json ]
then
    wget https://s3.amazonaws.com/visual-dialog/v0.9/visdial_0.9_val.zip
    unzip visdial_0.9_val.zip
    rm visdial_0.9_val.zip
    mv visdial_0.9_val.json visdial/
fi

# set up the coco folder and folders inside
mkdir -p coco
mkdir -p coco/images
mkdir -p coco/test
mkdir -p coco/train

# download COCO 2014 images for training and validation
if [[ $(ls coco/images | wc -l) != 123287 ]]
then
    wget http://images.cocodataset.org/zips/train2014.zip
    wget http://images.cocodataset.org/zips/val2014.zip
fi
if [ -f train2014.zip ]
then
    echo "Extracting training images. This might take a long while."
    unzip -qq -j train2014.zip -d coco/images/
    rm train2014.zip
    echo "    |-->Done"
fi
if [ -f val2014.zip ]
then
    echo "Extracting validation images. This might take a long while."
    unzip -qq -j val2014.zip -d coco/images/
    rm val2014.zip
    echo "    |-->Done"
fi

# download pre-trained caption embeddings
if [ ! -f coco/train/char-CNN-RNN-embeddings.pickle ]
then
    python google_drive.py 0B3y_msrWZaXLQXVzOENCY2E3TlU coco.zip
fi
if [ ! -f coco/test/val_captions.t7 ]
then
    python google_drive.py 0B3y_msrWZaXLeEs5MTg0RC1fa0U coco_test.zip
fi
if [ -f coco.zip ]
then
    echo "Extracting coco training caption embeddings from zip"
    unzip -qq coco.zip 'coco/train/filenames.pickle' -d .
    unzip -qq coco.zip 'coco/train/char-CNN-RNN-embeddings.pickle' -d .
    unzip -qq coco.zip 'coco/test/filenames.pickle' -d .
    unzip -qq coco.zip 'coco/test/filename.txt' -d .
    rm coco.zip
    echo "    |-->Done"
fi

if [ -f coco_test.zip ]
then
    echo "Extracting coco validation caption embeddings from zip"
    unzip -qq coco_test.zip 'test/val_captions.t7' -d coco/
    unzip -qq coco_test.zip 'test/val_captions.txt' -d coco/
    unzip -qq coco_test.zip 'test/val_filename.txt' -d coco/
    rm coco_test.zip
    echo "    |-->Done"
fi

echo "#############################################################################################"
echo "Now go to the code directory and execute"
echo "  CUDA_VISIBLE_DEVICES=0 python generate-dialogue-pickle.py --config cfg/coco_skipthought.yml"
echo "to generate skip-thought embeddings for the dialogs! The environment for running"
echo "skip-thoughts is not currently included."
echo "#############################################################################################"

