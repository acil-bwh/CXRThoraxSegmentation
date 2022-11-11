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
Definitive model is saved in ./models as **thorax_segmentation_model.h5**. Also other models are saved: **lung_segmentation_model.h5** and **mask_1.h5** used for loss. In ./models/classification_models are saved **lung_thorax_comparation_lung_model.h5** and **lung_thorax_comparation_thorax_model.h5**.

# DATA
## Segmented dataset
The NIH (National Institute of Health) Chest X-ray Dataset, comprises 112120 X-ray images with disease labels from 30805 unique patients. There are 15 classes (14 diseases and "No findings"). Images can be classified as "No findings" or one or more disease classes, showing 14 common thoracic pathologies. NIH Chest X-ray dataset itself does not contain lung field labels. 

For this project, 500 random images were selected from this dataset. The characteristics of this subset (**General Subset**) were the following: median age 48.5 years (IR 34-58.25 years), 206 (41%) females and 294 (59%) males. The different radiographic findings in this subset were: Atelectasis 49 (10%) Cardiomegaly 11 (2%), Consolidation 16 (3%), Edema 10 (2%), Effusion 64 (13%), Emphysema 20 (4%), Fibrosis 2 (0.4%), Hernia 1 (0.2%), Infiltration 97 (19%), Mass 33 (7%), No Finding 261 (52%), Nodule 26 (5%), Pleural Thickening 17 (3%), Pneumonia 10 (2%), Pneumothorax 36 (7%). This dataset was split into train and test, with an 80-20% proportion. 

In addition, another 200 images, with any type of pathology, were selected for testing, those without "No finding" label; in order to examine the segmentation capacity exclusively in pathologic thorax. The characteristics of this subset (**Pathologic Subset**) were as follows: median age 49 years (IR 34-58 years), 75 (37.5%) females and 125 (62.5%) males. The different findings in the radiographs were: Atelectasis 43 (22%), Cardiomegaly 11 (6%), Consolidation 13 (7%), Edema 11 (6%), Effusion 51 (26%), Emphysema 15 (8%), Fibrosis 5 (3%), Infiltration 82 (41%), Mass 27 (14%), Nodule 22 (11%), Pleural Thickening 17 (9%), Pneumonia 7 (4%), Pneumothorax 23 (12%). 

All images were segmented by a clinician. These images and masks datasets are in ***./data*** folder:

- General subset: ***./data/thorax_segmentation***, divided in train and test folders.

- Pathologic subset: ***./data/thorax_pathologic_segmentation***

## Clinical validation dataset 
Images from three different datasets were used for clinical validation. We selected 200 new random images from the NIH Chest X-ray dataset, the characteristics of this subset were as follow: median age of 49 years (IR 33-59 years), gender proportion of 94 (47%) females vs 106 (53%) males and 55% normal images (images labeled as “No finding”). Also, all images from the JSRT dataset were selected; this dataset, created by The Japanese Society of Radiological Technology, has 247 chest X-rays of 2048 × 2048 resolution, 38% images have no pathology, the median patient age is 60 years (50-69 years) and the gender proportion is 128 (52%) females vs 119 (48%) males. Finally, all images from The Montgomery County (MC) dataset were selected; this is a dataset created by the Department of Health and Human Services, Montgomery County, Maryland, USA, containing 138 CXR images, it has a 58% of normal images, the median patient age is 40 years (IR 28-51 years) and the gender proportion is 74 (54%) females vs 63 (46%) males.  

The resulting clinical validation dataset had 585 images, the median age was 51 years (IR 36.5-63 years), the gender proportion was 296 (51%) females vs 288 (49%) males and there was a 55% of normal radiographs. 

These validation images were introduced in the model, that outputted a mask. The resulting masks were applied over the original radiographs, and the segmented thoraxes were sent to four different clinicians. Each clinician gave a score between 1 and 3 to each segmentation, 1 was a perfect segmentation, 2 an acceptable segmentation but with some errors, and 3 an incorrect segmentation. All these images and masks are saved in ***./data/thorax_clinical_validation*** 

## Data augmentation
***./image_functions/image_fun.py***

Initially we had 400 training images (80% of the general dataset). To increase the amount of data we used "Albumentations" (https://github.com/albumentations-team/albumentations). To define an augmentation pipeline, it is needed to create an instance of the Compose class to which a list of augmentations to apply is passed (***albumentation()***). Data augmentation was applied twice to the train split of the general dataset, generating 800 new images in addition to the initial 400 (1200 total training images) (***augment_tensor()***). 

# PREPROCESSING
***./image_functions/image_fun.py***

The NIH images are in png format (1024, 1024, 3) so they were converted to grayscale. In addition, the size was reduced to 256 pixels, thus final images were (256, 256, 1). Finally, the images were normalized using z score (***create_tensor() -> normalize() -> read_img() -> recolor_resize()***)

# MODELS
For segmentation tasks U-Net is the most widely used network. That is why both networks that were tested had this architecture. Firstly, we used a very simple U-Net modality (**Simple Network**) with 40 layers and 34512193 trainable params. Secondly, it was used the U-Net referenced in the article by Wufeng Liu et al.30 (**Efficientnet Network**). The network architecture used in this work has five coding layers and five decoding layers. The encoder is Efficientnet-b4 pre-trained on the Imagenet49. The decoder consists of five blocks; each decoding layer includes a dropout layer, a two-dimensional convolution and padding layer, and finally, two residual blocks and a LeakyReLU. 

- Simple Network: ***model_functions/unet_funct.py*** and ***model_functions/unet_doble_loss.py***

- Efficientnet Network: ***model_functions/eff_unet.py***

# TRAINING 
***execute_thorax_training.py***

With Simple Network we used either Dice coefficient loss (Method 1, ***model_functions/unet_funct.py***) or a customized loss based on the publication of Ozan Oktay et al. (Method 2, ***model_functions/unet_doble_loss.py***). For Method 2 we initially trained a U-Net to generate an exact copy of a mask, this model is saved in ***./models/mask_1.h5***, and from this trained model we took just the encoder part. For the new loss, that we called Mask Loss (***MyLoss()***), we applied this encoder part over the output of the Simple Network, as the same time as it was applied over the ground truth mask. Both results were compared through an absolute difference. This Mask Loss is supposed to help the model in learning the thorax shape. The custom loss of Method 2 was the sum of loss mask and dice loss in a proportion of 0.6 vs 1 respectively.  

In Efficientnet Network Dice coefficient loss was used (Method 3, ***model_functions/eff_unet.py***). The paper by Wufeng Liu et al.30 does not specify how the fine tuning of the backbone was performed, so different tests were carried out. Initially, it was tried to leave the entire backbone blocked, which did not lead to good results, so layers were gradually unblocked until the entire blackbone was unblocked, thus achieving the best results. Therefore, when comparing the three methods, this latter option was the one used. 

In all methods optimizer Adam, with a learning rate of 1e-4, was used, since it was the one with which the best results were obtained. For training all the models, validation split of 0.2 with shuffle was applied. A batch of 8 images was used, with 200 epochs, on which early stopping with patience of 10 epochs was applied. 

After each training the resulting model is tested over the General Subset test split (100 images) (***./data/thorax_segmentation/test***) and over the Pathologic Subset (200 images) (***./data/thorax_pathologic_segmentation***) and results are saved in ***./results/thorax_segmentation/validation_results.csv*** and ***./results/thorax_segmentation/pathologic_validation_results.csv***. For the comparation between methods we use Dice coefficient, Mask Loss, accuracy, and AUC. Resulting models are saved in ***./models/model_name.h5***. The best results in training were achieved using Method 1 (***model_functions/unet_funct.py***), and the best model got using this method is saved in **./models/thorax_segmentation_model.h5**. 

# POSTPROCESSING
***./image_functions/mask_funct.py***

As the masks resulting from Method 1 and Method 2 had, in many cases, holes or extra fragments, post-processing was considered. To solve the fragments, the label() function from the measure module of scikit-image was applied over the mask, which labels connected regions, and only the largest region was retained . The binary_fill_holes() function from the ndimage module of the Scipy package, which fills the holes in binary objects, was used to solve the holes problem(***remove_parts()***). 

# CLASSIFICATION COMPARATION BETWEEN THORAX AND LUNG MAKS MODELS
## Lung segmentation model 
***execute_lung_training.py***

To prove the theory that thorax segmentation is better than lung segmentation models for lung segmentation were also trained with the same schema explained above. For these trainings, a dataset with 21165 images and its mask was used (https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset/data). This dataset has 10192 normal images, 3616 COVID images, 6012 images with lung opacity and 1345 pneumonia images (***./data/lung_segmentation***). Preprocessing was made in the explained way but no data augmentation was used; postprocessing was not apply. Trainig is made over the train dataset and after each training the resulting model is tested over the test dataset and results are saved in ***./results/lung_segmentation/validation_results.csv***. For the comparation between methods we use Dice coefficient, accuracy, and AUC. Resulting models are saved in ***./models/model_name.h5***. After training several models the best one was selected based on Dice coefficient and this is **./models/lung_segmentation_model.h5**. 

## Pathologic classification using both segmentations
***./classification_funct***

Lung and thorax segmentation models (**thorax_segmentation_model.h5** and **lung_segmentation_model.h5**) were compared using them as part of the image preprocessing (***./classification_funct/image/prepare_img.py***) for a classification model (normal vs pathologic).

### Training
***./classification_funct/image_funct***

***execute_classification_training.py***

We used 10000 images from the NIH dataset (5000 images labeled as No Finding and 5000 images with any pathological label), and they were split into train (80%) (***./data/pneumonia_classification/training_data***) and validation (29%) (***./data/pneumonia_classification/internal_validation***) folders. During the preprocessing of the images either the thorax or the lung mask was applied, followed by Clahe filter and normalization with z-score. Model results are saved in ***./results/classification/training_data.csv***. Ten trainings were made with each mask type and the best of each class was selected based on AUC score, they are saved in ***./models/classification_models***. 

### Validation
***./classification_funct/mask_funct***

***execute_classification_evaluation.py***

After selecting the best models both were applied over the validation folder (***./data/pneumonia_classification/internal_validation***) and also over an external validation dataset with pediatric images (normal and pneumonia) (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (***./data/pneumonia_classification/external_validation***). Internal and external validation results are saved in ***./results/classification/validation.csv***, also, plots from external validation are saved in a folder with the model name ***./results/classification/model_name/***