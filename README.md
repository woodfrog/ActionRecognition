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