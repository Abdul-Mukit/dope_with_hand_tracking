## Introduction
In this repository I have combined three functionalities to detect pose of target objects held by hand in dark.
Please watch my [demo video](https://youtu.be/XwVy5sZZxG8) for more details.

Following are the repositories/ideas I have relied on:
1. I have used DOPE [(Deep Object Pose Estimation)](https://github.com/NVlabs/Deep_Object_Pose) for tracking object pose. 
2. YOLO2 PyTorch impelementation by marvis ([pytorch-yolo2](https://github.com/marvis/pytorch-yolo2)) for tracking hands.
3. Gamma correction using OpenCV ([Link](https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/)) for processing
dark/low-exposure frames.

## Downloads
I uploaded the necessary weights in my [Dropbox](https://www.dropbox.com/sh/hv44h3v1zc21a2q/AADSNSIWrtf__8yKpqZnEUC4a?dl=0).

Put the 'backup' folder in the project direcory. 

Put the 'darknet19_448.conv.23' inside 'cfg' folder.

## Installation
Please follow instructions for [DOPE](https://github.com/NVlabs/Deep_Object_Pose).

The original pytorch-yolo2 was workable only with PyTorch version 0.3.1.
I have changed the codes so that it can work with 0.4.0 which DOPE uses. 

## Usage:
I have used an Intel Realsense D435i camera for this impelematation as I needed to control exposure.

live_dope_hand_realsense.py is the final impelmentation. In the code change the "Settings" section if needed.
Please watch my [demo video](https://youtu.be/XwVy5sZZxG8) for more details.

live_dope_realsense.py and live_dope_webcam.py are just DOPE demos using the realsense camera and webcam respectively.
The original DOPE demos are using ROS. I needed a smipler demo so I made these two.




