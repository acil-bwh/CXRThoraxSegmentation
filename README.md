# Steps to follow:

1. Clone the repository

```
git clone https://github.com/acil-bwh/CXRThoraxSegmentation.git
```
2. Instal requirements.txt
```
cd CXRThoraxSegmentation
pip install -r requirements.txt
```
3. Download datasets and models from (...). 

4. Save all datasets in the ./data repository folder and all the models in the ./model repository folder

5. Execute apply_mask_model giving your images dir as an argument. This will generate a folder in your dir named masks with all images masked.
```
python apply_mask_model.py -p ./your_images_folder_path
```
6. Execute any of the other execute_* to test the repository
```
python execute_thorax_training.py
python execute_lung_training.py
python execute_classification_training.py
python execute_classification_evaluation.py
```
Definitive model is saved as **./models/thorax_segmentation_model.h5**. Also other models are saved: **./models/lung_segmentation_model.h5** and **./models/mask_1.h5**, which is used for loss. Also **./models/classification_models/lung_thorax_comparation_lung_model.h5** and **./models/classification_models/lung_thorax_comparation_thorax_model.h5** are saved.

# DATA
## Thorax segmentation data
The NIH (National Institute of Health) Chest X-ray Dataset43 (https://www.kaggle.com/datasets/nih-chest-xrays/data), comprises 112120 X-ray images with disease labels from 30805 unique patients. There are 15 classes (14 diseases and "No findings"). Images can be classified as "No findings" or one or more disease classes, showing 14 common thoracic pathologies. NIH Chest X-ray dataset itself does not contain lung field labels. 

For this project, 500 random images were selected from this dataset (General Dataset). This dataset was split into train and test, with an 80-20% proportion. In addition, another 200 images, with any type of pathology, were selected for testing, those without "No finding" label; in order to examine the segmentation capacity exclusively in pathologic thorax.

All images were segmented by a clinician. These images and masks datasets are in ***./data*** folder:

- General Dataset: ***./data/thorax_segmentation/general_dataset***, divided in train and test folders.

- Pathologic Dataset: ***./data/thorax_segmentation/pathologic_dataset***

## Clinical validation dataset 
Images from three different datasets were used for clinical validation. We selected 200 new random images from the NIH Chest X-ray dataset, all images from the JSRT dataset (https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt) and all images from The Montgomery County (MC) dataset (https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery). The resulting Clinical Validation Dataset had 585 images, the median age was 51 years (IR 36.5-63 years), the gender proportion was 296 (51%) females vs 288 (49%) males and there was a 55% of normal radiographs. All these images and masks are saved in ***./data/thorax_segmentation/clinical_validation_dataset*** 

## Lung segmentation model
In order to create a lung segmentation model, a dataset with 21165 images and its mask was used (https://kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset). This dataset has 10192 normal images, 3616 COVID images, 6012 images with lung opacity and 1345 pneumonia images (Lung Segmentation Dataset). All these images and masks are saved in ***./data/lung_segmentation/lung_segmentation_dataset*** 

## Patologic classification model
Lung and thorax segmentation models were compared using them as part of the image preprocessing for a classification model (normal vs pathologic). To train this classification model 10000 images from the NIH dataset (5000 images labeled as No Finding and 5000 images with any pathological label) were used, and they were split into train (80%) and validation (29%) folders (Classification Dataset). After selecting the best models of each class (with lung mask preprocessing or with thorax mask preprocessing) both were applied over the validation folder and also over an External Classification Validation Dataset (https://kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset) with pediatric images (normal and pneumonia). 

All images were segmented by a clinician. These images and masks datasets are in ***./data*** folder:

- Classification Dataset: ***./data/classification/classification_dataset***, divided in train and test folders.

- External Classification Validation Dataset: ***./data/classification/external_classification_validation_dataset***

# CODING
## Data augmentation
***./image_functions/image_fun.py***

Initially we had 400 training images (80% of the General Dataset). To increase the amount of data we used "Albumentations" (https://github.com/albumentations-team/albumentations). To define an augmentation pipeline, it is needed to create an instance of the Compose class to which a list of augmentations to apply is passed (***./image_functions/image_fun.albumentation()***). Data augmentation was applied twice to the train split of the general dataset, generating 800 new images in addition to the initial 400 (1200 total training images) (***augment_tensor()***). 

## Preprocessing
***./image_functions/image_fun.py***

The NIH images are in png format (1024, 1024, 3) so they were converted to grayscale. In addition, the size was reduced to 256 pixels, thus final images were (256, 256, 1). Finally, the images were normalized using z score (***create_tensor() -> normalize() -> read_img() -> recolor_resize()***)

## Models
For segmentation tasks U-Net is the most widely used network. That is why both networks that were tested had this architecture. Firstly, we used a very simple U-Net modality (Simple Network) with 40 layers and 34512193 trainable params. Secondly, it was used the U-Net referenced in the article by Wufeng Liu et al.30 (Efficientnet Network). The network architecture used in this work has five coding layers and five decoding layers. The encoder is Efficientnet-b4 pre-trained on the Imagenet. The decoder consists of five blocks; each decoding layer includes a dropout layer, a two-dimensional convolution and padding layer, and finally, two residual blocks and a LeakyReLU. 

- Simple Network: ***./model_functions/unet_funct.py*** and ***./model_functions/unet_doble_loss.py***

- Efficientnet Network: ***./model_functions/eff_unet.py***

## Training 
***execute_thorax_training.py***

With Simple Network we used either Dice coefficient loss (Method 1, ***./model_functions/unet_funct.py***) or a customized loss based on the publication of Ozan Oktay et al. (Method 2, ***./model_functions/unet_doble_loss.py***). For Method 2 we initially trained a U-Net to generate an exact copy of a mask, this model is saved in **./models/mask_1.h5**, and from this trained model we took just the encoder part. For the new loss, that we called Mask Loss (***./model_functions/unet_doble_loss.MyLoss()***), we applied this encoder part over the output of the Simple Network, as the same time as it was applied over the ground truth mask. Both results were compared through an absolute difference. This Mask Loss is supposed to help the model in learning the thorax shape. The custom loss of Method 2 was the sum of loss mask and dice loss in a proportion of 0.6 vs 1 respectively.  

In Efficientnet Network Dice coefficient loss was used (Method 3, ***./model_functions/eff_unet.py***). The paper by Wufeng Liu et al.30 does not specify how the fine tuning of the backbone was performed, so different tests were carried out. Initially, it was tried to leave the entire backbone blocked, which did not lead to good results, so layers were gradually unblocked until the entire blackbone was unblocked, thus achieving the best results. Therefore, when comparing the three methods, this latter option was the one used. 

In all methods optimizer Adam, with a learning rate of 1e-4, was used, since it was the one with which the best results were obtained. For training all the models, validation split of 0.2 with shuffle was applied. A batch of 8 images was used, with 200 epochs, on which early stopping with patience of 10 epochs was applied. 

After each training the resulting model is tested over the General Subset test split (100 images) (***./data/thorax_segmentation/test***) and over the Pathologic Subset (200 images) (***./data/thorax_pathologic_segmentation***) and results are saved in ***./results/thorax_segmentation/validation_results.csv*** and ***./results/thorax_segmentation/pathologic_validation_results.csv***. For the comparation between methods we use Dice coefficient, Mask Loss, accuracy, and AUC. Resulting models are saved in ***./models/model_name.h5***. The best results in training were achieved using Method 1 (***./model_functions/unet_funct.py***), and the best model got using this method is saved in **./models/thorax_segmentation_model.h5**. 

## Postprocessing
***./image_functions/mask_funct.py***

As the masks resulting from Method 1 and Method 2 had, in many cases, holes or extra fragments, post-processing was considered. To solve the fragments, the label() function from the measure module of scikit-image was applied over the mask, which labels connected regions, and only the largest region was retained . The binary_fill_holes() function from the ndimage module of the Scipy package, which fills the holes in binary objects, was used to solve the holes problem(***./image_functions/mask_funct.remove_parts()***). 

## Lung segmentation model
***execute_lung_training.py***

To prove the theory that thorax segmentation is better than lung segmentation models for lung segmentation were also trained with the same schema explained above. Preprocessing was made in the explained way but no data augmentation was used; postprocessing was not apply. After training several models the best one was selected based on Dice coefficient. Trainig is made over the train dataset and after each training the resulting model is tested over the test dataset and results are saved in ***./results/lung_segmentation/validation_results.csv***. For the comparation between methods we use Dice coefficient, accuracy, and AUC. Resulting models are saved in **./models/model_name.h5**. After training several models the best one was selected based on Dice coefficient and this is **./models/lung_segmentation_model.h5**. 

## Pathologic classification model
As said, lung and thorax segmentation models (**./models/thorax_segmentation_model.h5** and **./models/lung_segmentation_model.h5**) were compared training and validating a classification model (normal vs pathologic). During the preprocessing of the images either the thorax or the lung mask was applied, followed by Clahe filter and normalization with z-score (***./classification_funct/image/prepare_img.py***). Adam optimizer vas used, with a learning rate of 10-4 and Binary Crossentropy was used as loss.  

### Training
***./classification_funct/image_funct***

***execute_classification_training.py***

We used 10000 images from the NIH dataset (5000 images labeled as No Finding and 5000 images with any pathological label), and they were split into train (80%) (***./data/pneumonia_classification/training_data***) and validation (29%) (***./data/pneumonia_classification/internal_validation***) folders. During the preprocessing of the images either the thorax or the lung mask was applied, followed by Clahe filter and normalization with z-score. Model results are saved in ***./results/classification/training_data.csv***. Ten trainings were made with each mask type and the best of each class was selected based on AUC score, they are saved in ***./models/classification_models***. 

### Validation
***./classification_funct/mask_funct***

***execute_classification_evaluation.py***

After selecting the best models both were applied over the validation folder (***./data/pneumonia_classification/internal_validation***) and also over an external validation dataset with pediatric images (normal and pneumonia) (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (***./data/pneumonia_classification/external_validation***). Internal and external validation results are saved in ***./results/classification/validation.csv***, also, plots from external validation are saved in a folder with the model name ***./results/classification/model_name/***