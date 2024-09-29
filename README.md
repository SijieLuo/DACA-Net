# **DACA-Net: Detail-aware network with contrast attention for locating liquid crystal display defects**

## **Introduction**
This repo contains the official PyTorch implementation of DACA-Net.


## **Highlight**

- Proposed a novel LCD defect detection network for better detection of tiny defects under low-contrast background.
- Refined CSPDarknet53 network to enhance detail awareness for tiny objects.
- Developed a low-level semantic deep fusion module to improving the detection performance for tiny objects.
- Proposed a dual-focus contrast enhancement attention module to improve low-contrast object detection.
- Made the first dataset for LCD defect detection available.

## **Installation**

Our codebase is based on [YOLOv5]([https://github.com/facebookresearch/detectron2](https://github.com/ultralytics/yolov5)). You only need to follow its instructions for installation.

## **Dataset Preparation**

### **LCD light defect dataset**

The LCD light defect dataset displayed defects on a 7-inch screen with a resolution of 768×1280, encompassing spot, line, and mura defects. The dataset is provided by the data fusion research team at the University of Electronic Science and Technology of China. To download the dataset, please visit:

Samples

![lcd_simples](https://github.com/user-attachments/assets/e5946377-02dd-4e98-8321-83884c2d0b23)

### **LCD surface defect dataset**

The surface defect dataset included three types of defects: oil, scratches and stains, with 400 images per defect type at a resolution of 1920×1080. The dataset is built and presented by Jian Zhang, Miaoju Ban (Open Lab on Human Robot Interaction, Peking University). To download the dataset, please visit: https://robotics.pkusz.edu.cn/resources/dataset/.

Samples

<p float="left">
  <img src="https://github.com/SijieLuo/Detail-aware-network-with-contrast-attention/assets/52660906/91ef6aa7-a645-4562-8274-2ae2c0174657" width="32%" />
  <img src="https://github.com/SijieLuo/Detail-aware-network-with-contrast-attention/assets/52660906/f371add6-8acc-4867-ab6a-af10aaf2bffa" width="32%" />
  <img src="https://github.com/SijieLuo/Detail-aware-network-with-contrast-attention/assets/52660906/0cfcba51-7b01-4e1a-819e-11bed5b57b81" width="32%" />
</p>

<p align="center">
  <span>oil</span>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<span>scratch</span>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<span>stain</span>
</p>

### **PCB surface defect dataset**

The PCB defect dataset contained 693 images with six types of defects: missing holes, open circuit, mouse bites, spur, short, and spurious copper. The dataset is built and presented by Lihui Dai et al. (Open Lab on Human Robot Interaction, Peking University). To download the dataset, please visit: https://robotics.pkusz.edu.cn/resources/dataset/.

Samples

![绘图2](https://github.com/SijieLuo/Detail-aware-network-with-contrast-attention/assets/52660906/bfa1176f-084a-4302-aa58-ea99bde3b24d)

## **Acknowledge**






