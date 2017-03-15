import numpy as np
import os, cv2, random

def combine_list_txt(list_dir):
    testlisttxt = ['testlist01.txt', 'testlist02.txt', 'testlist03.txt']
    trainlisttxt = ['trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt']

    testlist = []
    for fname in testlisttxt:
        txt_path = os.path.join(list_dir, fname)
        with open(txt_path) as fo:
            for line in fo:
                testlist.append(line.rstrip())

    trainlist = []
    for fname in trainlisttxt:
        txt_path = os.path.join(list_dir, fname)
        with open(txt_path) as fo:
            for line in fo:
                trainlist.append(line[:line.rfind(' ')])

    return trainlist, testlist

# extract video clip to 5 FPS, down sample resolution to 216*216, and make sequence length of 10
def process_clip(src_dir, dst_dir):
    SequenceLength = 10
    IMSIZE = (216, 216)

    cap = cv2.VideoCapture(src_dir)
    all_frames = []
    if (cap.isOpened()):
        while (True):
            ret, frame = cap.read()
            if ret:
                all_frames.append(frame)
            else:
                break
    else:
        print('Opening ', src_dir, ' fails')
        return

    clip_length = len(all_frames)
    if clip_length <= 20:
        print(src_dir, ' has no enough frames')
    step_size = int(clip_length / (SequenceLength + 1))
    frame_sequence = []
    index = random.randrange(step_size)  # generate random starting index
    for i in range(SequenceLength):
        frame = all_frames[index]
        frame = cv2.resize(frame, IMSIZE)
        # cv2.imshow('f', frame)
        # cv2.waitKey(30)
        frame_sequence.append(frame)
        index += step_size
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dst_dir, fourcc, 5.0, IMSIZE)
    for frame in frame_sequence:
        out.write(frame)

    out.release()
    cap.release()

def Preprocessing(list_dir, UCF_dir, dest_dir):
    if os.path.exists(dest_dir):
        return
    os.mkdir(dest_dir)
    trainlist, testlist = combine_list_txt(list_dir)
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    os.mkdir(train_dir)
    os.mkdir(test_dir)

    print('Processing train data')
    for clip in trainlist:
        clip_name = os.path.basename(clip)
        clip_category = os.path.dirname(clip)
        category_dir = os.path.join(train_dir, clip_category)
        src_dir = os.path.join(UCF_dir, clip)
        print(src_dir)
        dst_dir = os.path.join(category_dir, clip_name)
        if not os.path.exists(category_dir):
            os.mkdir(category_dir)
        process_clip(src_dir, dst_dir)

    print('Processing test data')
    for clip in testlist:
        clip_name = os.path.basename(clip)
        clip_category = os.path.dirname(clip)
        category_dir = os.path.join(test_dir, clip_category)
        src_dir = os.path.join(UCF_dir, clip)
        print(src_dir)
        dst_dir = os.path.join(category_dir, clip_name)
        if not os.path.exists(category_dir):
            os.mkdir(category_dir)
        process_clip(src_dir, dst_dir)



if __name__ == '__main__':
    data_dir = '/home/changan/ActionRocognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    UCF_dir = os.path.join(data_dir, 'UCF-101')
    dest_dir = os.path.join(data_dir, 'UCF-Preprocessed')

    Preprocessing(list_dir, UCF_dir, dest_dir)