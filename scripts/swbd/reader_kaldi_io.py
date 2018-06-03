import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset, DataLoader

import kaldi_io
import logging

logger = logging.getLogger(__name__)

'''
source: key feature_mat
target: key word_sequence

Note: should add <sos> and <eos> to the target sequence for enc-dec training
'''
def get_data_for_kaldi_io(source_rspecifier, target_rspecifier, bos=None, eos=None):
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

            if bos and eos:
                tgt_extend = np.concatenate(([bos], tgt[key], [eos]), axis=0).astype('int32')
            else:
                logger.warning('The target sequence has no <bos> and <eos> extend!')
                tgt_extend = tgt[key].astype('int32')

            tmp_label.append(tgt_extend)
            tmp_llength.append(len(tgt_extend))
            tmp_key.append(key)
            # cnt+=1
            # if cnt==10:
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
    def __init__(self, source_rspecifier, target_rspecifier, bos=None, eos=None):
        super(Reader, self).__init__()
        self.data, self.label, self.length, self.llength, self.utt_keys = get_data_for_kaldi_io(source_rspecifier, target_rspecifier, bos=bos, eos=eos)
        self.input_dim = self.data[0].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.length[index], self.llength[index], index

    def get_utt_key(self, index):
        return self.utt_keys[index]

    def get_valid_length(self):
        return [(l, ll) for l, ll in zip(self.length, self.llength)]
