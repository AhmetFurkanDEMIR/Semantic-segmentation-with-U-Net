![](https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white) ![](https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white) ![](https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white) 

# Semantic segmentation on earthquake data with U-net

![](https://static.dw.com/image/52147325_303.jpg)

Identifying the damaged parts of the buildings with the model we train on images of buildings damaged during the earthquake.

**Semantic segmentation**

![](https://www.mathworks.com/help/vision/ug/semanticsegmentation_transferlearning.png)

Semantic segmentation is a natural step in the progression from coarse to fine inference:The origin could be located at classification, which consists of making a prediction for a whole input.The next step is localization / detection, which provide not only the classes but also additional information regarding the spatial location of those classes.Finally, semantic segmentation achieves fine-grained inference by making dense predictions inferring labels for every pixel, so that each pixel is labeled with the class of its enclosing object ore region.


**U-net**

![](https://miro.medium.com/max/700/0*6bTOX4gO-mh8hLm2.png)

The U-Net architecture stems from the so-called “fully convolutional network” first proposed by Long, Shelhamer, and Darrell.

The main idea is to supplement a usual contracting network by successive layers, where pooling operations are replaced by upsampling operators. Hence these layers increase the resolution of the output. What's more, a successive convolutional layer can then learn to assemble a precise output based on this information.

One important modification in U-Net is that there are a large number of feature channels in the upsampling part, which allow the network to propagate context information to higher resolution layers. As a consequence, the expansive path is more or less symmetric to the contracting part, and yields a u-shaped architecture. The network only uses the valid part of each convolution without any fully connected layers. To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image. This tiling strategy is important to apply the network to large images, since otherwise the resolution would be limited by the GPU memory. 

### **Data and our model**

There are 1560 damage images (earthquakes) as data and 4 masks of each image. We did a training with U-net on these images, then we asked for pixel estimates over the data it had never encountered.

We used a wide network structure, we used the relu activation function in the intermediate layers and the sigmoid activation function in the last layer. In the last layer, we made a binary classification, so is the pixel value 1 or 0. The value 1 represents damaged pixels, the value 0 represents the smooth and stable pixels. We used Adam as the optimization algorithm and binary_crossentropy as the loss function. batch_size=128, epochs=175

![untitled-f001190](https://user-images.githubusercontent.com/54184905/111334058-4c5e8500-8684-11eb-8820-adbba779b119.png)

As can be seen in the test picture above, the damaged parts, the parts with 1, are white, the undamaged parts, that is, the parts with 0, are black.

#### **[Test Video](/Test.mp4)**

##### **References**

[Link1](https://www.kaggle.com/abdulkarimx2/keras-u-net-starter-lb-0-277)

[Link2](https://medium.com/@pallawi.ds/semantic-segmentation-with-u-net-train-and-test-on-your-custom-data-in-keras-39e4f972ec89)

[Link3](https://github.com/neptune-ai/open-solution-mapping-challenge)

