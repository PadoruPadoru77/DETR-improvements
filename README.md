# DETR-Improvements

The DETR (DEtection TRansformer) model is an end-to-end object detection framework that reformulates detection as a direct set prediction problem using a transformer architecture. Unlike traditional detectors that rely on hand-crafted components such as anchor boxes, region proposals, and non-maximum suppression (NMS), DETR uses a convolutional backbone to extract image features and a transformer encoder–decoder to reason globally about object relationships. A fixed number of learnable object queries are passed to the decoder, each of which predicts a single object’s class and bounding box, with training supervised through bipartite (Hungarian) matching between predictions and ground-truth objects. While this design greatly simplifies the detection pipeline and enables strong global reasoning, DETR suffers from several limitations: it converges slowly and often requires hundreds of training epochs, struggles with small objects due to coarse feature resolution, and is sensitive to class imbalance—where frequent (head) classes dominate the matching and learning process, causing rare (tail) classes to be under-detected or misclassified. These challenges have motivated numerous extensions and improvements to enhance training efficiency and robustness.

This project will address the problems with DETR model using a modified version of the PASCAL VOC dataset, as well as exploring improvements to it. 
Some techniques that will be used includes implementing Denoising DETR, lowering the number of decoder layers and queries, and implementing Automatic Mixed Precision training. Overall the highest accuracy with the IoU threshold of 0.5 achieved was 0.608%, with an improved inference speed of 0.0582s and number of parameters reduced to 38,474,714.
Feel free to checkout my report and presentation for more details! 

**Note:** I've also included the final code I've written to achieve my performance, as well as the dataset, collab notebook, and final trained model. The dataset, final model, and model used for initial training are found below:
Data: https://drive.google.com/file/d/1LjkrqlCuNAhwV3yFGEOt0AzZj4lqUPR_/view?usp=sharing
Final Model: https://drive.google.com/file/d/1nFxZB06vyVzhkY9s8BXKBWmHOpmPUVjY/view?usp=sharing
Model used for training: https://drive.google.com/file/d/1MuY1ZMWpNzp3gpPgxws2sb5mAO96tn_v/view?usp=sharing

<p align="center"> 
  <img width="1136" height="352" alt="image" src="https://github.com/user-attachments/assets/1627335a-6677-4168-b65f-d092398db64a" />
  <img width="502" height="202" alt="image" src="https://github.com/user-attachments/assets/6d1b8f02-8bd5-409d-b8d6-79ea2576ebcb" />
  <img width="1008" height="623" alt="image" src="https://github.com/user-attachments/assets/35aaccb5-64d5-410d-ac5c-0e6c820f929c" />
  <img width="779" height="325" alt="image" src="https://github.com/user-attachments/assets/32d06124-528e-40a7-8374-97ce108c46a2" />
  <img width="382" height="223" alt="image" src="https://github.com/user-attachments/assets/e63289f3-49e1-4758-b761-91bb699de4ca" />
</p>


