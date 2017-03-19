from utils.model_processing import model_processing
from models.CNN import CNN



model = CNN(include_top=False)
print('Loaded Inception model without top layers')
print(model.summary())
src_dir = '/home/changan/ActionRocognition_rnn/data/UCF-Preprocessed'
dest_dir = '/home/changan/ActionRocognition_rnn/data/CNN_Processed'
TIMESEQ_LEN = 10
model_processing(model, src_dir, dest_dir, TIMESEQ_LEN)