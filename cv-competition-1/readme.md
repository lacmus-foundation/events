# 1st Lacmus Comouter Vision Competition (July 2021)
## Overview

## The Task Definition
During the competition, you are invited to solve the *one class object detection* task. 

### Taraining data
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

The implementation of these metrics can be found in the file `metrics.py`

### Baseline

We have 2 baselines: with pytorch and tensorflow

- Pytorch Jupyter Notebook
- Tensorflow Jupyter Notebook

### How to get better results?
- try to pretrain model on Stenford Drone Dataset or VisDrone DET dataset
- try augumentations
- try other network architectures