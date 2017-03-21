import numpy as np
import scipy.misc
import os, cv2, random


def combine_list_txt(list_dir):
    testlisttxt = 'testlist01.txt'
    trainlisttxt = 'trainlist01.txt'

    testlist = []
    txt_path = os.path.join(list_dir, testlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            testlist.append(line.rstrip())

    trainlist = []
    txt_path = os.path.join(list_dir, trainlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            trainlist.append(line[:line.rfind(' ')])

    return trainlist, testlist

# down sample image resolution to 216*216, and make sequence length 10
def process_clip(src_dir, dst_dir, seq_len, img_size, mean=None):
    all_frames = []
    cap = cv2.VideoCapture(src_dir)
    while cap.isOpened():
        succ, frame = cap.read()
        if not succ:
            break
        # append frame that is not all zeros
        if frame.any():
            all_frames.append(frame)
    # save all frames
    if seq_len is None:
        all_frames = np.stack(all_frames, axis=0)
        dst_dir = os.path.splitext(dst_dir)[0] + '.npy'
        np.save(dst_dir, all_frames)
    else:
        clip_length = len(all_frames)
        if clip_length <= 20:
            print(src_dir, ' has no enough frames')
        step_size = int(clip_length / (seq_len + 1))
        frame_sequence = []
        index = random.randrange(step_size)  # generate random starting index
        for i in range(seq_len):
            frame = all_frames[index]
            frame = scipy.misc.imresize(frame, img_size)
            frame = frame.astype(dtype='float16')
            if mean is not None:
                frame -= mean
            frame /= 255
            frame_sequence.append(frame)
            index += step_size
        frame_sequence = np.stack(frame_sequence, axis=0)
        dst_dir = os.path.splitext(dst_dir)[0]+'.npy'
        np.save(dst_dir, frame_sequence)

    cap.release()

def Preprocessing(list_dir, UCF_dir, dest_dir, seq_len, img_size, mean_subtraction=True):
    if os.path.exists(dest_dir):
        print('Destination directory already exists')
        return
    os.mkdir(dest_dir)
    trainlist, testlist = combine_list_txt(list_dir)
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    if mean_subtraction:
        mean = calc_mean(UCF_dir, img_size).astype(dtype='float16')
        np.save(os.path.join(dest_dir, 'mean.npy'), mean)

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
        if mean_subtraction:
            process_clip(src_dir, dst_dir, mean=mean, seq_len=seq_len, img_size=img_size)
        else:
            process_clip(src_dir, dst_dir, seq_len=seq_len, img_size=img_size)

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
        if mean_subtraction:
            process_clip(src_dir, dst_dir, mean=mean, seq_len=seq_len, img_size=img_size)
        else:
            process_clip(src_dir, dst_dir, seq_len=seq_len, img_size=img_size)


def calc_mean(UCF_dir, img_size):
    frames = []
    print('Calculating RGB mean ...')
    for dirpath, dirnames, filenames in os.walk(UCF_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if os.path.exists(path):
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    # successful read and frame should not be all zeros
                    if ret and frame.any():
                        if frame.shape != (240, 320, 3):
                            frame = scipy.misc.imresize(frame, (240, 320, 3))
                        frames.append(frame)
                cap.release()
    frames = np.stack(frames)
    mean = frames.mean(axis=0, dtype='int64')
    mean = scipy.misc.imresize(mean, img_size)
    print('RGB mean is calculated over', len(frames), 'video frames')
    return mean


if __name__ == '__main__':
    '''
        extract frames from videos as npy files
    '''
    sequence_length = 10
    image_size = (216, 216, 3)

    data_dir = '/home/changan/ActionRocognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    UCF_dir = os.path.join(data_dir, 'UCF-101')
    seq_frames_dir = os.path.join(data_dir, 'UCF-Preprocessed')
    all_frames_dir = os.path.join(data_dir, 'UCF_all_frames')

    # extract all frames from videos to train the network
    # Preprocessing(list_dir, UCF_dir, all_frames_dir, seq_len=None, img_size=image_size,
    #               mean_subtraction=True)

    # only extract 10 frames from each video to train the network
    Preprocessing(list_dir, UCF_dir, seq_frames_dir, seq_len=sequence_length, img_size=image_size, mean_subtraction=True)