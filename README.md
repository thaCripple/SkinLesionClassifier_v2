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
