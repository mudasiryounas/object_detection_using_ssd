# Object detection using SSD with pre trained model


###Libraries:
imageio        2.9.0  
imageio-ffmpeg 0.4.2  
numpy          1.19.2  
opencv-python  4.4.0.44  
torch          1.6.0  
torchvision    0.7.0  

## What is SSD?
SSD is a method for detecting objects in an image using single neural network (thats why its called sibgle shot).
SSD detects positions of the objects by guessing and then calculating the erros, and if not satosfied then backpropagate to update
results and try again.


**imageio:** a library to read frames from original video and then write predicted frames on a new video


The already trained model 'ssd300_mAP_77.43_v2.pth' we are using in here is trained to detect 30-40 objects
**Reference:** https://github.com/amdegroot/ssd.pytorch





asd


# Demo

<p align="center"><img src="https://github.com/mudasiryounas/object_detection_using_ssd/blob/master/demo1-output.gif" width="550"></p>

<p align="center"><img src="https://github.com/mudasiryounas/object_detection_using_ssd/blob/master/demo2-output.gif" width="550"></p>


