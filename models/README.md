### The models we have tried:

1. Simple CNN + LSTM, basic idea is from the paper of [Long-term recurrent
   convolutional network](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf)

2. Fine-tuning Residual Network (the version of ResNet50). With the keras implementation
   from [keras repo](https://github.com/fchollet/deep-learning-models)
   
3. Two-stream CNN, combine the **spatial model of 2** with another CNN which accepts optical flow as input


