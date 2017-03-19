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
1. Try ConvLSTM. Train from sratch, too slow and seems no effect.
2. Seperate Inception and LSTM, and only train LSTM. Loss stops dropping after 100 epochs.
3. Train Inception using video frame data. And then only train LSTM using output of Inception.
Finding loss not decreasing, try data normalization(/255). Still fail, so we guess
this may because CNN con only recognize the rough outline of an object, but can not
tell the small difference of what that obejct is doing.
4. Use small CNN and RNN and train them seperately. To prevent overfitting, add 
regularizer in FC layer, dropout layer with 0.5 dropping rate, setting checkpoint
to save the weights when validation accuracy reaches highest. Tranining small CNN hundreds of
epochs make trainning acc 98, testing acc 28. Still heavy overfitting.

Methods to try: 
1. Use two stream method, add input of optical flow.
2. Data augumentation, when sampling, generating more videos
3. mean subtracion along with normalization