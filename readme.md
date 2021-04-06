# Real-time Multi-object Pose estimation
+ This is my undergraduate project in 2019 to implement real-time mutli-object pose estimation in real-time with RGB images as input
+ The project inspired by Microsofy Research Project:https://github.com/microsoft/singleshotpose and YOLOv3
+ Original trained network weight has lost due to my moving and graduation
+ The repo will be updated and with new network weight at next version
+ Chosen results in my undergraduate thesis
  + Single Object\
    ![img](imgs/single-add.png)
  + Multi Object\
    ![gt](imgs/gt.png)![pred](imgs/pred.png)
    (left:groundtruth,right:prediction)
+ To do
  + [ ] Re-orgnize and re-implement the codes
  + [ ] Add more augmentation tricks
  + [ ] Transfer to YCB-Dataset or other dataset (Orginally train/test on Linemod and Linemod-occlusion)
  + [ ] Improve accuracy
+ Data
  + internal matrix
    + fx=572.41140, px=325.26110, fy=573.57043; py=242.04899