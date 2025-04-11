# SkinLesionClassifier_v2

## I learn how to create utility functions, store them in a module on github and import them as needed
I created several helper functions and classes that I reuse throughout the workflow. I learned how to create my own python package and download only select files and folders from a repository.

In the *auxiliary* package I saved my own custom ImageDataset class based on the torchvision's version. It has extended functinality (retrieving only a portion of the samples, distinguishing between original and augmented images) and it works with pytorch DataLoader.

Very happy about that one :grinning:

## I learn about **oversampling** in the *slc_v2_traindata_oversampling* notebook
Since the dataset is quite imbalanced I use image augmentation to create more training samples for the model and use those in conjunction with class weights in the loss function.
![augmented images](https://github.com/user-attachments/assets/c7c742c7-47d1-4f60-b24b-2066a5424bae)
![training_samples_distribution](https://github.com/user-attachments/assets/13041712-e5ca-407a-b581-a8b3ffa0a276)

## I run an experiment on several models in the *slc_v2-model_selection_experiments* notebook
I compare 4 model architectures' performance on the dataset. This was intended as a Tensorboard learning experience but unfortunately it didn't work in Colab, so instead it was a reminder and practice of working with multi-indexed pandas dataframes :+1:
![experiment_history](https://github.com/user-attachments/assets/0a2ec44d-1066-4da8-814b-39c56606db13)

## I fine-tune EfficientNet_b0, b3 and MaxViT_t on and oversampled [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
I learned so much in this section! Thanks to the gained experience in terms of optimizing batch sizes, handling large files and discovering `lightning.Fabric` I was able to train 3 different models and fully compare their performance.

I only include one notebook - *maxvit_t_slc_v2_TheModel.ipynb* because they were all very similar

## Next I compare the performance of the 3 models in the *slc_v2-model_comparison.ipynb* notebook
This was a great exercise in working with DataFrames with hierarchical indexes and using them with `matplotlib` and `seaborn`!

I was incredibly suprised at how close the models were in terms of accuracy and recall. I think it's because the test data was very similar to the training samples despite even the augmentations applied.

I don't expect the stellar 99+% accuracy to hold up in the real world since the model will be classifying smartphone photos not dermatoscopic images, but the high number is still nice to see 😃

Given the collected data I decide to pick the **EfficientNet_b3** architecture for my application. The file size is still relatively small and it has the best performance, albeit marginally 😆 

It also achieves prediction speed well below the treshold of ~5-10 seconds I had set at the outset.
![metrics](https://github.com/user-attachments/assets/d542fa23-1061-4333-90b5-dac4ecc68fe2)
![sizes](https://github.com/user-attachments/assets/cc7acf7c-538b-4436-9212-1115b4c2d49a)
![times](https://github.com/user-attachments/assets/97a1fe72-32e0-40f2-93f7-37e3d7ef571f)

