# ece285-final-project
Final Project - ECE 285

### Description

This is project **Object Detection in the Autonomous Driving Scenario** developed by team *The Object Detectors* composed of Anwesan Pal, David Paz-Ruiz and Harshini Rajachander. 

Object Detection in the Autonomous Driving Scenario is a project intended to explore state of the art architectures for real-time multi-object detection for autonomous vehicle detection modules and then attempt to classify the cars into further sub-categories decided by their respective make. 

### Requirements
Install packages:
$ pip install --user -r requirements.txt 

### Code organization
- [Github Repository](https://github.com/AftermathK/ece285-final-project) 
- Drive Link: Sent via email. 

1. Our Entire Pipeline:  
    1. git clone https://github.com/AftermathK/ece285-final-project.git
    2. cd ece285-final-project
    2. Download weights for Yolo from this [link](https://pjreddie.com/media/files/yolov3.weights) and save it in the current directory.
    3. Download the model weights for the Resnet18 classifier from the drive link. 
    3. Download the images for Vehicle classificaier from the drive link and place it in the repositary's main directory. 
    4. Run demo_final.ipynb
  
2. Faster R-CNN: (demo file)
    1. cd Faster R-CNN
    2. Follow instructions given in readme file. 
    3. faster_rcc_demo.ipynb - Run main demo file for viewing performance of faster rcnn on single image/video files. 
  
3. RetinaNet
    1. Follow instructions in the given readme.
    
4. Experiments folder:
    1. project_train_(Resnet18).ipynb : Run the training for our Vehicle Classifier. 
    2. project_train_cars_demo.ipynb  : Run training on experimental networks of VGG16 and Resnet18.
    3. project_train_cars_demo-(AlexVgg19).ipynb : Run training on experimental networks of VGG19 and AlexNet.
 
