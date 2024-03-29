# DATA
## Thorax segmentation data
The NIH (National Institute of Health) Chest X-ray Dataset43 (https://www.kaggle.com/datasets/nih-chest-xrays/data), comprises 112120 X-ray images with disease labels from 30805 unique patients. There are 15 classes (14 diseases and "No findings"). Images can be classified as "No findings" or one or more disease classes, showing 14 common thoracic pathologies. NIH Chest X-ray dataset itself does not contain lung field labels. 

For this project, 500 random images were selected from this dataset (General Dataset). This dataset was split into train and test, with an 80-20% proportion. In addition, another 200 images, with any type of pathology, were selected for testing, those without "No finding" label; in order to examine the segmentation capacity exclusively in pathologic thorax.

## Clinical validation dataset 
Images from three different datasets were used for clinical validation. We selected 200 new random images from the NIH Chest X-ray dataset, all images from the JSRT dataset (https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt) and all images from The Montgomery County dataset (https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery). The resulting Clinical Validation Dataset had 585 images, the median age was 51 years (IR 36.5-63 years), the gender proportion was 296 (51%) females vs 288 (49%) males and there was a 55% of normal radiographs.

## Lung segmentation model
In order to create a lung segmentation model, a dataset with 21165 images and its mask was used (https://kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset). This dataset has 10192 normal images, 3616 COVID images, 6012 images with lung opacity and 1345 pneumonia images (Lung Segmentation Dataset). 

## Patologic classification model
Lung and thorax segmentation models were compared using them as part of the image preprocessing for a classification model (normal vs pathologic). To train this classification model all NIH dataset was used. It was split into train 66% and test 33%. All PA images from PADCHEST dataset were used for external validation of the model. As the External Validation Dataset did not share the same labels as the training dataset, we took steps to unify them. We conducted a search for all NIH labels and identified atelectasis, cardiomegaly, consolidation, emphysema, mass, nodule, pneumonia, pleural thickening, and pneumothorax. For the remaining missing labels, we matched effusion with loculated fissural effusion, loculated pleural effusion, pericardial effusion, and pleural effusion. We matched edema with pulmonary edema, fibrosis with pulmonary fibrosis, hernia with hiatal hernia, infiltration with infiltrates, and no finding with normal. After this process, 26,546 images were left with no label.
To study pathology classification gender bias, all NIH dataset was used but it was split in two depending on the gender (Gender Bias Female Dataset and Gender Bias Male Dataset), and those two datasets were split again in train 66% and test 33%. 


# CODING
## Segmentation models
### Thorax segmentation model
***./image_functions/image_fun.py***

Initially we had 400 training images (80% of the General Dataset). To increase the amount of data we used "Albumentations" (https://github.com/albumentations-team/albumentations). To define an augmentation pipeline, it is needed to create an instance of the Compose class to which a list of augmentations to apply is passed (***./image_functions/image_fun.albumentation()***). Data augmentation was applied twice to the train split of the general dataset, generating 800 new images in addition to the initial 400 (1200 total training images) (***augment_tensor()***). 


***./image_functions/image_fun.py***

The NIH images are in png format (1024, 1024, 3) so they were converted to grayscale. In addition, the size was reduced to 256 pixels, thus final images were (256, 256, 1). Finally, the images were normalized using z score (***create_tensor() -> normalize() -> read_img() -> recolor_resize()***)

For segmentation tasks U-Net is the most widely used network. That is why both networks that were tested had this architecture. Firstly, we used a very simple U-Net modality (Simple Network) with 40 layers and 34512193 trainable params. Secondly, it was used the U-Net referenced in the article by Wufeng Liu et al.30 (Efficientnet Network). The network architecture used in this work has five coding layers and five decoding layers. The encoder is Efficientnet-b4 pre-trained on the Imagenet. The decoder consists of five blocks; each decoding layer includes a dropout layer, a two-dimensional convolution and padding layer, and finally, two residual blocks and a LeakyReLU. 

- Simple Network: ***./model_functions/unet_funct.py*** and ***./model_functions/unet_doble_loss.py***

- Efficientnet Network: ***./model_functions/eff_unet.py***


***execute_thorax_training.py***

With Simple Network we used either Dice coefficient loss (Method 1, ***./model_functions/unet_funct.py***) or a customized loss based on the publication of Ozan Oktay et al. (Method 2, ***./model_functions/unet_doble_loss.py***). For Method 2 we initially trained a U-Net to generate an exact copy of a mask, this model is saved in **./models/mask_1.h5**, and from this trained model we took just the encoder part. For the new loss, that we called Mask Loss (***./model_functions/unet_doble_loss.MyLoss()***), we applied this encoder part over the output of the Simple Network, as the same time as it was applied over the ground truth mask. Both results were compared through an absolute difference. This Mask Loss is supposed to help the model in learning the thorax shape. The custom loss of Method 2 was the sum of loss mask and dice loss in a proportion of 0.6 vs 1 respectively.  

In Efficientnet Network Dice coefficient loss was used (Method 3, ***./model_functions/eff_unet.py***). The paper by Wufeng Liu et al.30 does not specify how the fine tuning of the backbone was performed, so different tests were carried out. Initially, it was tried to leave the entire backbone blocked, which did not lead to good results, so layers were gradually unblocked until the entire blackbone was unblocked, thus achieving the best results. Therefore, when comparing the three methods, this latter option was the one used. 

In all methods optimizer Adam, with a learning rate of 1e-4, was used, since it was the one with which the best results were obtained. For training all the models, validation split of 0.2 with shuffle was applied. A batch of 8 images was used, with 200 epochs, on which early stopping with patience of 10 epochs was applied. 

After each training the resulting model is tested over the General Subset test split (100 images) (***./data/thorax_segmentation/test***) and over the Pathologic Subset (200 images) (***./data/thorax_pathologic_segmentation***) and results are saved in ***./results/thorax_segmentation/validation_results.csv*** and ***./results/thorax_segmentation/pathologic_validation_results.csv***. For the comparation between methods we use Dice coefficient, Mask Loss, accuracy, and AUC. Resulting models are saved in ***./models/model_name.h5***. The best results in training were achieved using Method 1 (***./model_functions/unet_funct.py***), and the best model got using this method is saved in **./models/thorax_segmentation_model.h5**. 


***./image_functions/mask_funct.py***

As the masks resulting from Method 1 and Method 2 had, in many cases, holes or extra fragments, post-processing was considered. To solve the fragments, the label() function from the measure module of scikit-image was applied over the mask, which labels connected regions, and only the largest region was retained . The binary_fill_holes() function from the ndimage module of the Scipy package, which fills the holes in binary objects, was used to solve the holes problem(***./image_functions/mask_funct.remove_parts()***). 

### Lung segmentation model
***execute_lung_training.py***

To prove the theory that thorax segmentation is better than lung segmentation models for lung segmentation were also trained with the same schema explained above. Preprocessing was made in the explained way but no data augmentation was used; postprocessing was not apply. After training several models the best one was selected based on Dice coefficient. Trainig is made over the train dataset and after each training the resulting model is tested over the test dataset and results are saved in ***./results/lung_segmentation/validation_results.csv***. For the comparation between methods we use Dice coefficient, accuracy, and AUC. Resulting models are saved in **./models/model_name.h5**. After training several models the best one was selected based on Dice coefficient and this is **./models/lung_segmentation_model.h5**. 

## Pathologic classification model
***./classification_masks/***

The aim of classification model was to differentiate between normal and pathological images.
Original images from NIH in (1024,1024,3) were transformed to (512,512,1). Then either lung mask, thorax mask or no mask were applied, followed by clahe filter and normalization with z-score.

The classification model was built on a pretrained Xception backbone network that was originally trained on ImageNet. To adapt gray-scale images, a 2D convolution of dimensions (1,1,3) was added as a preceding layer to the backbone. The backbone was then followed by another 2D convolution of dimensions (1,1,3000), and a global max pooling followed by four dense layers. 
During training, the first half of the backbone was kept frozen while the second half was fine-tuned. Binary Crossentropy was used as the loss function with Adam optimizer and a learning rate of 10-4.

Twenty trainings were made with each mask preprocessing, so 60 trainings in total. All of them were validated over the internal and External Validation Dataset, and each preprocessing method was compared using AUC.

### Gender Bias
Based on Larrazabal et al. paper we designed an experiment where we try to check how biases produced by gender imbalances were modified by segmentation. 

With each train dataset (Gender Bias Male Dataset and Gender Bias Female Dataset), twenty trainings were made with each mask preprocessing, so 60 trainings for each dataset, 120 trainings in total. Each model was validated over both male and female validation datasets and results were compared using AUC.
