import numpy as np
import random, cv2

SequenceLength = 10
IMSIZE = (216, 216)

src_dir = '/home/changan/ActionRocognition_rnn/data/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
cap = cv2.VideoCapture(src_dir)
all_frames = []
if(cap.isOpened()):
    while(True):
        ret, frame = cap.read()
        if ret:
            all_frames.append(frame)
        else:
            break
else:
    print('Opening ', src_dir, ' fails')
    #return

clip_length = len(all_frames)
if clip_length <= 20:
    print(src_dir, ' has no enough frames')
step_size = int(clip_length/(SequenceLength+1))
frame_sequence = []
index = random.randrange(step_size)# generate random starting index
for i in range(SequenceLength):
    frame = all_frames[index]
    frame = cv2.resize(frame, IMSIZE)
    # cv2.imshow('f', frame)
    # cv2.waitKey(30)
    frame_sequence.append(frame)
    index += step_size
print(len(frame_sequence))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('makeup5.avi', fourcc, 5.0, IMSIZE)
for frame in frame_sequence:
    out.write(frame)

out.release()
cap.release()
cv2.destroyAllWindows()
