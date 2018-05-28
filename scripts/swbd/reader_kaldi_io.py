import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset, DataLoader

import kaldi_io
import logging

logger = logging.get_logger(__name__)

'''
source: key feature_mat
target: key word_sequence

Note: should add <sos> and <eos> to the target sequence for enc-dec training
'''
def get_data_for_kaldi_io(source_rspecifier, target_rspecifier):
    src = {}
    tgt = {}
    for key,mat in kaldi_io.read_mat_ark(source_rspecifier):
        src[key] = mat
    for key,vec in kaldi_io.read_vec_int_ark(target_rspecifier):
        tgt[key] = vec

    tmp_data = []
    tmp_label = []
    tmp_length = []
    tmp_llength = []
    tmp_key = []
    cnt=0
    for key, feature in src.items():
        if tgt.get(key) is not None:
            tmp_data.append(feature)
            tmp_length.append(len(feature))

            tgt_extend = np.concatenate(([tgt_vocab['<bos>']], tgt[key], [tgt_vocab['<eos>']]), axis=0)
            tmp_label.append(tgt_extend)
            tmp_llength.append(len(tgt_extend))
            tmp_key.append(key)
            # cnt+=1
            # if cnt==100:
            #     break
        else:
            logger.warning("{:s} has no label".format(key))
            pass

    return tmp_data, tmp_label, tmp_length, tmp_llength, tmp_key

'''
data: shape (max_length, feature_size)
label: shape (max_llength, )
'''
class Reader(Dataset):
    def __init__(self, source_rspecifier, target_rspecifier):
        super(Reader, self).__init__()
        self.data, self.label, self.length, self.llength, self.utt_keys = get_data_for_kaldi_io(source_rspecifier, target_rspecifier)
        self.input_dim = self.data[0].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.length[index], self.llength[index], index

    def get_utt_key(self, index):
        return self.utt_keys[index]
