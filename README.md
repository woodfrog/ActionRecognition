# Action recognition (detection) based on RNN

### Overall Objective:

- Training, validating, testing RNN models (with LSTM cells) using data from open-source video datasets.

- Make the model give reasonable descriptions with word or sentence when it's
  fed with new videos.


### File Structure

./rnn\_practice: 
    For doing some practice on RNN models and LSTMs with online tutorails and
    other great resources.

./data:
    Training and testing data. (**But don't add huge data files to this repo**)

./model:
    Defining the model and other training/testing details

./utils:
    Possible utils scripts

### Pipeline
#### Preprocessing
1. Download dataset: UCF101 http://crcv.ucf.edu/data/UCF101.php  
2. Extract video to 5 FPS and down sample resolution for each video 
and discard videos with too few frames
3. Segment number of frames into equal size blocks(frame number/sequence 
length L). Randomly select one frame from each block to compose L length 
video clip
4. Load videos in batch and do mean substraction for each video
5. Feed preprocessed videos into CNN and train on RNN with LSTM network

#### Experiments
1. Try ConvLSTM. Train from sratch, too slow and seems no effect with acc around 0.01, the probability of random guess.
2. Seperate Inception and LSTM, and only train LSTM. Loss stops dropping after 100 epochs.
3. Train Inception using video frame data. And then only train LSTM using output of Inception.
Finding loss not decreasing, try data normalization(/255). Still fail, so we guess
this may because CNN con only recognize the rough outline of an object, but can not
tell the small difference of what that obejct is doing.
4. Use small CNN and RNN and train them seperately. To prevent overfitting, add 
regularizer in FC layer, dropout layer with 0.5 dropping rate, setting checkpoint
to save the weights when validation accuracy reaches highest. Tranining small CNN hundreds of
epochs make trainning acc 98, testing acc 28. Still heavy overfitting. Training RNN, 
training acc stops at 0.33, validation acc stops at 0.17
5. Build a more complex CNN model according (two stream ...), try to extracts all frames to get more data,
 but fail to fit the data in disk memory.
7. Simply using a combination of Resnet as feature extractor and one layer lstm gives val acc of 0.67
8. Regenerate data with mean subtraction and normalization.Fine tune ResNet with one more FC layer and get val acc of 0.59. 
Using a combination of finetuned ResNet and lstm shows no improvement. 
This indicates that RNN is not so useful in action recognition, since it only keeps incoming 
information in its state but does not explicit operations on coming sequence. In other words, action recognition
is not a task that depends on long term dependence so much, but instead, it needs more explicit information like optical
flow
9. Try ConvNet using optical flow as input, and get val acc of 0.21. Using continuous sequential data and higher drop rate
(0.5 to 0.9)



Methods to try: 
1. Data augumentation, random sampling, generate different preprocessed data
2. Using continuous frames as input 
3. Multi-task learning: combine different databases using two different softmax and shared weights
4. fliiping and jittering, loss decreasing stops soon?