# Cmpe297_Sec98_EmergyingTechnologies_Project

Brain Tumor Detection

Objective:

Brain tumor occurs because of anomalous development of cells. It is one of the major reasons of death in adults around the globe. Millions of deaths can be prevented through early detection of brain tumor. Earlier brain tumor detection using Magnetic Resonance Imaging (MRI) may increase patient's survival rate. In MRI, tumor is shown more clearly that helps in the process of further treatment. This project aims to detect tumor at an early phase.The main purpose of the project is to detect if a person has brain tumor or not using their MRI scans. For this purpose we have implemented different model architectures to understand and perform the detection.

Dataset Description:

Brain MRI images are used as the input image of count tumor images 260 and no tumor images of 100. It consists of mri scans of two classes
YES - tumor, encoded as 1.
NO - no tumor, encoded as 0.

![MRI Scan with Tumor](/Images/Y17)


![MRI Scan with No Tumor](/Images/17 no)

Data Augmentation

As the input dataset size is comparatively smaller, data augmentation technique is performed to increase the traing set data. And also data augmentation helps in avoiding over training of the model. 


![Some of the data augmented images](/Images/DataAugmentedImages)

Models implemented:

1. VGG16 and VGG16 with Transfer Learning

It usually refers to a deep convolutional network for object recognition developed and trained by Oxford's renowned Visual Geometry Group (VGG), which achieved very good performance on the ImageNet dataset.
The VGG-16 model is a 16-layer (convolution and fully connected) network built on the ImageNet database, which is built for the purpose of image recognition and classification.

2. MobileNet

TensorFlow is an open-source library for numeric computation using dataflow graphs. It was developed by Google brain team as a proprietary machine learning system based on deep learning neural networks.
We used a special class of convolutional neural networks called MobileNets. They are optimized to be executed using minimal possible computing power.

3. ResNet

Identity shortcut connection that skips one or more layers. Artificial neural network (ANN) of a kind that builds on constructs known from pyramidal cells in the cerebral cortex. Residual neural networks do this by utilizing skip connections, or shortcuts to jump over some layers. Typical ResNet models are implemented with double- or triple- layer skips that contain nonlinearities (ReLU) and batch normalization in between.

4. CNN

CNNs are powerful image processing, artificial intelligence (AI) that use deep learning to perform both generative and descriptive tasks, often using machine vison that includes image and video recognition.

Other Features Implemented are:

Transfer Learning

Transfer learning is used for improvization and keep generalization rather than memorization.

Model Analyses

Brain tumor is one of the most critical and highly risky disease. And we have analysed the brain MRI scan images using different neural network models to understand and predict if a person has tumor or not. Accuracy achieved through Convolutional neural network is 87%, VGG is 80%, VGG16 Transfer Learning is 81%, MobileNet is 85% and ResNet is 75%.

