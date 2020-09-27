# Object detection using SSD


**Libraries:**  pytorch, imageio

## What is SSD?
SSD is a method for detecting objects in an image using single neural network (thats why its called sibgle shot).
SSD detects positions of the objects by guessing and then calculating the erros, and if not satosfied then backpropagate to update
results and try again.


**imageio:** a library to read frames from original video and then write predicted frames on a new video


The already trained model 'ssd300_mAP_77.43_v2.pth' we are using in here is trained to detect 30-40 objects
**Reference:** https://github.com/amdegroot/ssd.pytorch