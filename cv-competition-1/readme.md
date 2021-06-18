# 1st Lacmus Computer Vision Competition (July 2021)
## Overview
Lacmus
Finding Missing People With Neural Networks.
Fast. User-Friendly. OpenSource.

###Problem: Missing Person
Every person going outdoor, has a chance to get lost.
Usually people get lost in the forest and wilderness. 
About 120 000 Persons are lost In Russia every year.
More than 100 000 Persons are lost in the USA every year.

Usually person, who lost in wilderness, meets a number of risk factors, such as: Dehydration, Hypothermia, Trauma, Panic
Unfortunately, this person has a chance to die without medical assistance.
A classical way to find a person, who get lost in wilderness, is a Search Operation with Ground Search & Rescue Squads.
This way takes a great chance to find a lost person, but it is really slow. Also, this kind od Search Operation depends on human resources. 
More people in operation make search faster.
How can we boost the search? We can use copters to make photos of the area, where some person get lost. 
After that we can analyze this photos and trying to found a photo with missing person. Good way!
But.. Analyzing photos by eyes also needs human resources.
How can we speed up this process? We can use Neural Networks! Convolutional Neural Networks process a large number of photos extremly faster, than human.
Let's do it!
Our project, Lacmus, helps Search & Rescue Squads to find people, who get lost, by analyzing photos from copters with Neural Networks.
And now we are ready to Competition!
It is a classical one class object detection task.
We are need to detect people on photos from copters with estimated altitude about 50m (164 ft). 

## The Task Definition
During the competition, you are invited to solve the *one class object detection* task. 

### Training data
The dataset consists of pictures and annotations in Pascal VOC format winh 1 class - `Pedestrian`:

```xml
<annotation>
    <folder>VocGalsTfl</folder>
    <filename>0</filename>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>4000</width>
        <height>3000</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>Pedestrian</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>1881</xmin>
            <ymin>1409</ymin>
            <xmax>1905</xmax>
            <ymax>1469</ymax>
        </bndbox>
    </object>
    <object>
        ...
    </object>
</annotation> 
```

The structure of the data:
```
├── EmptyPart
│   └── JPEGImages
│       └── X.jpg
└── TrainingData
    ├── Annotations
    │   └── X.xml
    ├── ImageSets
    │   └── Main 
    │       # *.txt which split the dataset
    │       ├──  test.txt
    │       ├──  train.txt
    │       ├──  trainval.txt
    │       └──  val.txt
    ├── JPEGImages
    │   └── X.jpg
    └── Description.pdf
```

### Metrics
We considering two metrics:
- mAP @ IoU 50
- F1 @ IoU 50

The implementation of these metrics can be found in the file [`metrics.py`](pytorch_baseline/metrics.py)

### Baseline

We have 2 baselines: with pytorch and tensorflow

- [Pytorch Jupyter Notebook](pytorch_baseline/pytorch_baseline.ipynb)
- [Tensorflow Jupyter Notebook](tf_baseline/tf_baseline.ipynb)

### How to get better results?
- try to pretrain model on [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) or VisDrone DET dataset
- try augumentations
- try other network architectures
