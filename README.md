# Action Recognition

### Overview

- Explore some action recognition models


-  Dataset: [UCF-101](http://crcv.ucf.edu/data/UCF101.php) 
    

- Compare the performance of different models and do some analysis based on the experiment results



### File Structure of the Repo

rnn\_practice: 
    For doing some practice on RNN models and LSTMs with online tutorials and
    other useful resources

data:
    Training and testing data. (**But don't add huge data files to this repo, add them to gitignore**)

models:
    Defining the architecture of models

utils:
    Utils scripts for dataset preparation, input pre-processing and other misc  
    
train_CNN:
    For training the different CNN models. Load corresponding model, set the training parameters and then start training 

process_CNN:
    For the LRCN model, the CNN component is pre-trained and then fixed during the training of LSTM cells. Thus we can use the 
    CNN model to pre-process the frames of each video and store the intermediate result for feeding into LSTMs later. This 
    can largely improve the training efficiency of the LRCN model

train_RNN:
    For training the LRCN model
   
predict:
    For calculating the overall testing accuracy on the whole testing set
    
 

### Models Description

- Fine-tuned ResNet50 trained solely with single-frame image data (every frame of every
   video is considered as an image for training or testing, thus a natural data augmentation).
   The ResNet50 is from [keras repo](https://github.com/fchollet/deep-learning-models), with weights 
   pre-trained on Imagenet. **./models/finetuned_resnet.py** 
   
![](https://github.com/woodfrog/ActionRecognition/blob/master/readme_imgs/finetuned_resnet.png?raw=true)   

- LRCN (CNN feature extractor(here we use the fine-tuned ResNet50) + LSTMs). Input is a sequence of frames uniformly extracted from each video. The fine-tuned ResNet directly uses the result of 1 without extra training (C.F.[Long-term recurrent
   convolutional network](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf)).
   
   **Produce intermediate data using ./process_CNN.py and then train and predict with ./models/RNN.py**
    
![](https://github.com/woodfrog/ActionRecognition/blob/master/readme_imgs/LRCN.png?raw=true)
   
   
- Simple CNN model trained with stacked optical flow data (generate one stacked optical flow from each of the video, and use the optical flow as the input of the network). **./models/temporal_CNN.py**
   
![](https://github.com/woodfrog/ActionRecognition/blob/master/readme_imgs/CNN_optical_flow.png?raw=true)

- Two-stream model, combine the models in 2 and 3 with an extra fusion layer that
   output the final result. 3 and 4 refer to [this paper](http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf)
   **./models/two_stream.py**

![](https://github.com/woodfrog/ActionRecognition/blob/master/readme_imgs/two_stream_model.png?raw=true)
