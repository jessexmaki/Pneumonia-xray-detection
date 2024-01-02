# Pneumonia Lung X-Ray Detection

This project aims to develop a deep learning model to detect pneumonia from lung X-ray images. We use a convolutional neural network (CNN) built on the TensorFlow and Keras frameworks for this task.

## Dataset
We utilized a dataset consisting of 176 images, divided into training and validation sets with 136 and 40 images respectively.

## Model Architecture
The model architecture is built using MobileNetV2 as a base, with the following layers:
- Rescaling and Data Augmentation layers for preprocessing
- Functional MobileNetV2 layer pretrained on ImageNet
- MaxPooling and Flatten layers
- Dense layer for binary classification (pneumonia vs. normal)

## Training
The model was trained for 20 epochs. We observed the following:
- The training accuracy started at 41.89% and improved to 99.32%.
- The validation accuracy started at 65.00% and achieved up to 95.00%.

  <img width="968" alt="Screenshot 2024-01-02 at 5 44 59 PM" src="https://github.com/jessexmaki/pneumonia-xray-detection/assets/87655161/6b10b765-52b3-4eda-90c1-bb5ce8b72773">
  
     <img width="823" alt="Screenshot 2024-01-02 at 5 45 33 PM" src="https://github.com/jessexmaki/pneumonia-xray-detection/assets/87655161/acd0e384-7315-4a69-83bc-59015d55f829">

## Challenges and Considerations
- Class imbalance in the dataset might lead to biased predictions. Techniques like oversampling, undersampling, or using class weights during training can be considered to mitigate this.
- Further tuning and regularization methods could be explored to improve the model's generalization on unseen data.

## Usage
To use the model, provide an X-ray image, and the model will predict whether the image indicates pneumonia. Ensure the images are preprocessed (rescaled and resized) as per the model's requirements.

## Dependencies
- TensorFlow
- Keras
- NumPy
- Matplotlib (for visualization)

## Conclusion
This project demonstrates the potential of deep learning in medical imaging, providing a tool that could assist in early and quick pneumonia detection. Further improvements and validations with a larger, more diverse dataset are necessary for real-world applications.


