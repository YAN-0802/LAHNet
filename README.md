# Iris-LAHNet: A Lightweight Attention-guided High-resolution Network for Iris Segmentation and Localization
## Introduction
This is the code for IRIS-LAHNET. Iris-LAHNet is composed of a stem, a basic backbone, Pyramid Dilated Convolution (PDC) blocks, Cascaded Attention-guided Feature Fusion Module (C-AGFM), and auxiliary heads. The basic backbone is a tiny high-resolution network. The introduction of PDC blocks and C-AGFM helps to extract multi-scale features from multi-resolution images and reduce noise. In addition, we introduce three auxiliary heads with edge heatmaps, which output auxiliary loss to help model training and enhance attention to single pixels of the edge. It helps to compensate for the neglect of localization tasks during multi-task training. Experiments on four datasets show that our model achieves the lightest while ensuring segmentation and localization results.


## Requirements
