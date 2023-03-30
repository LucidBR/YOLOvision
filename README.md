# YOLOvision: Object Detection and Instance Segmentation AI project

YOLOvision is an AI project that includes state-of-the-art object detection and instance segmentation models. The models
are part of the YOLO (You Only Look Once) family, which is a group of models that use deep learning to perform real-time
object detection, classification, and localization.

## Object Detection Models

The object detection models in YOLOvision can detect and locate objects in real-time images and video streams. These
models can be trained to detect specific types of objects, such as cars, people, animals, or any other object of
interest. The YOLO models are known for being fast and accurate, making them a good choice for real-time applications.

## Instance Segmentation Models

The instance segmentation models in YOLOvision can detect and locate individual objects within an image or video frame.
Unlike object detection models, instance segmentation models also provide a mask for each object, which can be used to
separate them from the background. This makes them ideal for applications like medical imaging, where precise
segmentation is essential.

## YOLO Family

The YOLO family includes several models, each with their own strengths and weaknesses. YOLOv1 was the first model in the
family, and it introduced the concept of object detection using a single neural network. YOLOv2 improved upon the
original model by incorporating a more powerful backbone network and making it easier to train. YOLOv3 further improved
on the model by incorporating new techniques such as feature pyramid networks and improved training strategies.

## Customizable Models

With YOLOvision, users can create their own custom models by modifying the configuration files and training them with
their own datasets. This allows users to build models tailored to their specific needs and applications.

## Special Models

In addition to the standard object detection and instance segmentation models, YOLOvision also includes models for other
types of computer vision tasks. For example, there is a model for determining the age and gender of people in images, as
well as a model for detecting emotions.

## Overall

YOLOvision is a powerful AI project that provides a variety of high-quality models for object detection and
instance segmentation tasks. The models are part of the YOLO family, which is known for being fast and accurate, and
they can be easily customized to meet the needs of specific applications.

# Note

A Part of the idea and source code is from YOLO v8 ultralytics and YOLOvision is an edited version of YOLO v8
(a weak version for education and self learning) and CLI is not perfect (it's not working) if you want to use this
project you are free
this project is [MIT](https://github.com/erfanzar/YOLOvision/blob/main/LICENSE.md) licenced, but it's recommended to
use [ultralytics](https://github.com/ultralytics/ultralytics) for more stability

## Usage Note

YOLOvision currently share same api as ultralytics YOLOv8 so you can predict use and train model via same api as
ultralytics

### Example

```python
## Use normal detection model 
from YOLOvision import YOLO

# to build model from yaml file just pass yaml file (same as ultralytics)
model = YOLO('MODEL.pt')
image = 'your image (path , url , numpy array ...)'
predicts = model.predict(image, save=True)

```

Segmentation

```python
## Use Segmentation model 
from YOLOvision import YOLO, prepare_segmentation

# to build model from yaml file just pass yaml file (same as ultralytics)
model = prepare_segmentation(YOLO('MODEL-Segmentation.pt'))
image = 'your image (path , url , numpy array ...)'
predicts = model.predict(image, save=True)

```

If you want to use segmentation models you have to set the task manually for v 0.0.1 models

```python
from YOLOvision import YOLO, prepare_segmentation

model = prepare_segmentation(YOLO('YOLOvision-Segmentaion-M.pt'))
## Ready to Use 
```