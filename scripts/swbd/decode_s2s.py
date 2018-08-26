"""
Neural Machine Translation model for ASR
=================================

This example shows swbd 300h dataset E2E ASR

"""

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import time
import socket
import random
import os
import io
import logging
import kaldi_io

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from mxnet.gluon.data import DataLoader

import gluonE2EASR.data.batchify as btf
from gluonE2EASR.data.sampler import FixedBucketSampler
from gluonE2EASR.model import BeamSearchScorer
from gluonE2EASR.log import setup_main_logger, log_mxnet_version
from gluonE2EASR.vocab import Vocab

from simple_encoder_decoder import get_simple_encoder_decoder
from model import NMTModel, BeamSearchTranslator
from loss import SoftmaxCEMaskedLoss
from bleu import compute_bleu
from wer import compute_wer

from reader_kaldi_io import Reader, TestReader
import utils

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

############################################
########## Arguments specify ###############
############################################
parser = argparse.ArgumentParser(description='SwithBorad 300h End-to-End ASR Example.'
                                             'We train the Google NMT model')

parser.add_argument('--src_sym', type=str, default=None, help='[optional] source vocab')
parser.add_argument('--tgt_sym', type=str, default=None, help='target vocab.'
                                    'reserved words:'
                                    '\'<bos>\': begin of sentance; \'<eos>\': end of sentance;'
                                    '\'<unk>\': unknown token; \'<pad>\': padding token')

parser.add_argument('--test_src_rspecifier', type=str, help='training feature specifier(with kaldi scp/ark format)')

parser.add_argument('--init_model', type=str, default='', help='the initail model other than randon initailize')
parser.add_argument('--cell', type=str, default='lstm', help='the cell type [ lstm | gru ]')
parser.add_argument('--attention', type=str, default='dot', help='the cell type [ scaled_luong, normed_mlp, dot ]')
parser.add_argument('--hidden_size', type=int, default=300, help='Dimension of the embedding vectors and states.')
parser.add_argument('--dropout', type=float, default=0, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--num_enc_layers', type=int, default=3, help='number of layers in the encoder')
parser.add_argument('--num_dec_layers', type=int, default=3, help='number of layers in the decoder')
parser.add_argument('--enc_bidirectional', default=False, action='store_true', help='enc_bidirectional')
parser.add_argument('--dec_bidirectional', default=False, action='store_true', help='dec_bidirectional')

parser.add_argument('--beam_size', type=int, default=1, help='Beam size')
parser.add_argument('--lp_alpha', type=float, default=1.0, help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')

parser.add_argument('--num_gpus', type=int, default=None, help='number of the gpu to use. Set it to empty means to use cpu.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_buckets', type=int, default=5, help='Bucket number')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the throughput of the bucketing')
parser.add_argument('--tgt_max_len', type=int, default=100, help='Maximum length of the target sentence')

parser.add_argument('--save_dir', type=str, default='exp_debug', help='directory path to save the final model and training log')
############################################
########## Arguments handle  ###############
############################################
args = parser.parse_args()
if args.save_dir is None:
    args.save_dir = os.path.join(os.getcwd(), name)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

decode_dir = os.path.join(args.save_dir, 'decode')
if not os.path.exists(decode_dir):
    os.makedirs(decode_dir)

log_file = os.path.join(decode_dir, 'decode.log')
if os.path.exists(log_file):
    os.remove(log_file)

logger = setup_main_logger(name=__name__, path=log_file)
log_mxnet_version(logger)
logger.info("hostname: {}".format(socket.gethostname()))
logger.info(args)
############################################
############### Data loading ###############
############################################

logger.info("=============================")
logger.info("Loding the test data...")
logger.info("=============================")

if args.src_sym is not None:
    src_vocab = Vocab(vocab_file=args.src_sym, unknown_token=None, padding_token='<blk>',
                                              bos_token=None, eos_token=None)
if args.tgt_sym is not None:
    tgt_vocab = Vocab(vocab_file=args.tgt_sym, padding_token='<blk>')

logger.info("Test set:\n source:{}\n ".format(args.test_src_rspecifier))
data_test = TestReader(args.test_src_rspecifier)

data_test_lengths = data_test.get_valid_length()

test_batchify_fn = btf.Tuple(btf.Pad(axis=0), btf.Stack(), btf.Stack())

test_batch_sampler = FixedBucketSampler(lengths=data_test_lengths,
                                         batch_size=args.batch_size,
                                         num_buckets=args.num_buckets,
                                         ratio=args.bucket_ratio,
                                         shuffle=False)
logger.info('Test Batch Sampler:\n{}'.format(test_batch_sampler.stats()))
test_data_loader = DataLoader(data_test,
                               batch_sampler=test_batch_sampler,
                               batchify_fn=test_batchify_fn,
                               num_workers=1)

if args.num_gpus is None:
    ctx = [mx.cpu()]
else:
    ctx = [mx.gpu(i) for i in range(args.num_gpus)]
logger.info(ctx)
############################################
############ Model construction ############
############################################
encoder, decoder = get_simple_encoder_decoder(cell_type=args.cell,
                                            attention_cell=args.attention,
                                            hidden_size=args.hidden_size,
                                            dropout=args.dropout,
                                            num_enc_layers=args.num_enc_layers,
                                            num_dec_layers=args.num_dec_layers,
                                            enc_bidirectional=args.enc_bidirectional,
                                            dec_bidirectional=args.dec_bidirectional)
# for 1-of-k or posterior input
src_embed = gluon.nn.HybridSequential(prefix='src_embed_')
with src_embed.name_scope():
    src_embed.add(gluon.nn.Dense(args.hidden_size, in_units=len(src_vocab),
                                  weight_initializer=mx.init.Uniform(0.1), flatten=False))

model = NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 src_embed=src_embed, embed_size=args.hidden_size, prefix='gnmt_')

logger.info(model.collect_params().keys())

if args.init_model == '':
    logger.error("no model specified !")
else:
    model.load_parameters(args.init_model, ctx=ctx)
# model.hybridize()

logger.info(model)

translator = BeamSearchTranslator(model=model, beam_size=args.beam_size,
                                  scorer=BeamSearchScorer(alpha=args.lp_alpha, K=args.lp_k),
                                  max_length=args.tgt_max_len)
logger.info('Use beam_size={}, alpha={}, K={}'.format(args.beam_size, args.lp_alpha, args.lp_k))

############################################
############# Decoding process #############
############################################
translation_out = []
all_inst_ids = []
eos_int = tgt_vocab.token_to_idx[tgt_vocab.eos_token]

for batch_id, (src_seq, src_valid_length, inst_ids) in enumerate(test_data_loader):

    xpu_src_seq = utils.split_and_load(src_seq, ctx)
    xpu_src_valid_length = utils.split_and_load(src_valid_length, ctx)
    #### multi gpus ###
    # Translate
    for xpu_X, xpu_XL in zip(xpu_src_seq, xpu_src_valid_length):
        samples, _, sample_valid_length =\
            translator.translate(src_seq=xpu_X, src_valid_length=xpu_XL)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    
    # for xpu_X, xpu_XL in zip(xpu_src_seq, xpu_src_valid_length):
    #     batch_size = xpu_X.shape[0]

    #     enc_mems, enc_states = model.encode(inputs=xpu_X, valid_length=xpu_XL)
    #     decoder_states = model.decoder.init_state_from_encoder(enc_mems, enc_states, xpu_XL)
    #     step_inputs = mx.nd.full(shape=(batch_size,), ctx=xpu_X.context, dtype=np.float32,
    #                               val=tgt_vocab.token_to_idx[tgt_vocab.bos_token])
        
    #     tmp_tans = [ [] for _ in range(batch_size) ]
    #     break_flag = mx.nd.full(shape=(batch_size,), dtype=np.int32, val=0)
    #     for i in range(args.tgt_max_len):
    #         step_outputs, decoder_states, _ = model.decode_step(step_inputs, decoder_states)
    #         trans = mx.ndarray.argmax(step_outputs, axis=1) # step_outputs shape (batch_size, vocab_size)
            
    #         for j in range(batch_size):
    #             curr = int(trans[j].asscalar())
    #             if curr == eos_int:
    #                 break_flag[j] = 1
    #             if break_flag[j] != 1:
    #                 tmp_tans[j].append(tgt_vocab.idx_to_token[curr])
            
    #         if break_flag.sum().asscalar() == batch_size:
    #             break
    #         step_inputs = trans.as_in_context(step_inputs.context)

    #     translation_out.extend(tmp_tans)

    ### end of multi gpus ###
    all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
    logger.info('[Batch {}/{}]'.format(batch_id+1, len(test_data_loader)))
    
test_translation_out = [None for _ in range(len(all_inst_ids))]
for ind, sentence in zip(all_inst_ids, translation_out):
    test_translation_out[ind] = sentence

assert len(test_translation_out) == len(data_test), 'the # of translation shoule be same as the test set'

trans_file = os.path.join(decode_dir, 'result.txt')
with io.open(trans_file, 'w', encoding='utf-8') as of:
    for i in range(len(data_test)):
        sent = ' '.join(test_translation_out[i])
        key = data_test.get_utt_key(i)
        of.write(key + ' ' + sent + '\n')

