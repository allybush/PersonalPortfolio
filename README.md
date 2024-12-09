# Personal Portfolio
Included are files of my work on an AI image classification project with a short description of their function.

[accuracy.py](https://github.com/allybush/PersonalPortfolio/blob/d9b7617b56fa4054e46f392dd90658e0784af300/accuracy.py) Created attention-based heat maps (example [here](https://drive.google.com/file/d/1fqt1eQlRrsc5l8lG_roOdWqL6XbJzIry/view?usp=sharing)) for the Xception CNN architecture using the [Grad-CAM method] (https://doi.org/10.48550/arXiv.1610.02391). Saved heat map for each image, in addition to the confusion matrix for the model's predictions, to a folder.  

[trainmodel.py](https://github.com/allybush/PersonalPortfolio/blob/d9b7617b56fa4054e46f392dd90658e0784af300/trainmodel.py) Created a convolutional neural network using the Keras Python library and implemented transfer learning on several datasets. Adjusted parameters to prevent overfitting in fine-tuning stage by adding weight decay, learning rate decay, and extensive dropout.
