# Cmpe297_Sec98_EmergyingTechnologies_Project

Brain Tumor Detection

Objective:

Brain tumor occurs because of anomalous development of cells. It is one of the major reasons of death in adults around the globe. Millions of deaths can be prevented through early detection of brain tumor. Earlier brain tumor detection using Magnetic Resonance Imaging (MRI) may increase patient's survival rate. In MRI, tumor is shown more clearly that helps in the process of further treatment. This project aims to detect tumor at an early phase.The main purpose of the project is to detect if a person has brain tumor or not using their MRI scans. For this purpose we have implemented different model architectures to understand and perform the detection.

Dataset Description:

Brain MRI images are used as the input image of count tumor images 260 and no tumor images of 100. It consists of mri scans of two classes
YES - tumor, encoded as 1.
NO - no tumor, encoded as 0.

![MRI Scan with Tumor](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/Y17.jpg)


![MRI Scan with No Tumor](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/17%20no.jpg)

Data Augmentation

As the input dataset size is comparatively smaller, data augmentation technique is performed to increase the traing set data. And also data augmentation helps in avoiding over training of the model. 


![Some of the data augmented images](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/DataAugmentedImages.JPG)

Models implemented:

1. VGG16 and VGG16 with Transfer Learning

It usually refers to a deep convolutional network for object recognition developed and trained by Oxford's renowned Visual Geometry Group (VGG), which achieved very good performance on the ImageNet dataset.
The VGG-16 model is a 16-layer (convolution and fully connected) network built on the ImageNet database, which is built for the purpose of image recognition and classification.

![Vgg16 Architecture](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/Vgg16Architecture.JPG)

![Vgg16 Model Summary](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/Vgg16_ModelSummary.JPG)

2. MobileNet

TensorFlow is an open-source library for numeric computation using dataflow graphs. It was developed by Google brain team as a proprietary machine learning system based on deep learning neural networks.
We used a special class of convolutional neural networks called MobileNets. They are optimized to be executed using minimal possible computing power.

![MobileNet Architecture](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/MobileNetArchitecture.JPG)

![MobileNet Model Summary](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/MobileNet_ModelSummary.JPG)

3. ResNet

Identity shortcut connection that skips one or more layers. Artificial neural network (ANN) of a kind that builds on constructs known from pyramidal cells in the cerebral cortex. Residual neural networks do this by utilizing skip connections, or shortcuts to jump over some layers. Typical ResNet models are implemented with double- or triple- layer skips that contain nonlinearities (ReLU) and batch normalization in between.

![ResNet Architecture](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/ResNetArchitecture.JPG)

![ResNet Model Summary](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/ResNet_ModelSummary.JPG)

4. CNN

CNNs are powerful image processing, artificial intelligence (AI) that use deep learning to perform both generative and descriptive tasks, often using machine vison that includes image and video recognition.

![CNN Model Summary](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/CNN_ModelSummary.JPG)

Implemented our model in TensorBoard

![Tensorboard](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/TensorBoard_CNN1.jpg)
![Tensorboard of model](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/TensorBoard_CNN2.jpg)

Implemented our project with AutoML and analyses the metrics

![Tensorboard](https://github.com/yagnapriyad/Cmpe297_Sec98_EmergyingTechnologies_Project/blob/master/Images/AutoML.jpeg)

Other Features Implemented are:

Transfer Learning

Transfer learning is used for improvization and keep generalization rather than memorization.

Result

Brain tumor is one of the most critical and highly risky disease. And we have analysed the brain MRI scan images using different neural network models to understand and predict if a person has tumor or not. Accuracy achieved through Convolutional neural network is 87%, VGG is 80%, VGG16 Transfer Learning is 81%, MobileNet is 85% and ResNet is 75%.

Analyses

The  results  depict  that  automatic  detection  of  brain  tumor  can  be  done  more  confirmly  from  the  MRI  images  in  comparisons  to  other  tumor  detection systems available in the market. Further improvements also can be proposed with huge amount of MRI image data if available.

Extra Credit

1. Created an amazon s3 bucket to model the training data.

2. Created amazon sagemaker training instance to run and mangae jupyter notebook.

3. Processed data and training the model.

4. Deployed the model in amazon sage maker.

5. Validated data and generated accuracy predictions.

References:

1. https://link.springer.com/chapter/10.1007/978-3-642-20998-7_38

2. https://ieeexplore.ieee.org/document/7725009

3. https://ieeexplore.ieee.org/document/7583949



