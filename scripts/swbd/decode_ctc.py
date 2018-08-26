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
import random
import socket
import os
import io
import logging
import kaldi_io

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import DataLoader

import gluonE2EASR.data.batchify as btf
from gluonE2EASR.data.sampler import FixedBucketSampler
from gluonE2EASR.log import setup_main_logger, log_mxnet_version
from gluonE2EASR.vocab import Vocab

from model import Nnet

from reader_kaldi_io import Reader, TestReader
import kaldi_io
import utils

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

############################################
########## Arguments specify ###############
############################################
parser = argparse.ArgumentParser(description='SwithBorad 300h End-to-End ASR Example.'
                                             'We train the Google NMT model')
parser.add_mutually_exclusive_group(required=False)

parser.add_argument('--src_sym', type=str, default=None, help='[optional] source vocab')
parser.add_argument('--tgt_sym', type=str, default=None, help='target vocab.')
parser.add_argument('--test_src_rspecifier', type=str, help='test data feature specifier(with kaldi scp/ark format)')
parser.add_argument('--apply_exp', default=False, action='store_true', help='apply the exp for input feature')

parser.add_argument('--init_model', type=str, default='', help='the initail model other than randon initailize')

parser.add_argument('--num_layers', type=int, default=3, help='number of layers in the encoder')
parser.add_argument('--cell', type=str, default='lstm',
                    help='the cell type [ lstm | bilstm | gru ]')
parser.add_argument('--hidden_size', type=int, default=300, help='Dimension of the embedding '
                                                                'vectors and states.')
parser.add_argument('--embed_dropout', type=float, default=0,
                    help='dropout applied to embedding layers (0 = no dropout)')
parser.add_argument('--cell_dropout', type=float, default=0,
                    help='dropout applied to RNN cell layers (0 = no dropout)')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_buckets', type=int, default=5, help='Bucket number')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the '
                                                                    'throughput of the bucketing')

parser.add_argument('--save_dir', type=str, default='exp_debug',
                    help='directory path to save the final model and training log')
parser.add_argument('--num_gpus', type=int, default=None,
                    help='number of the gpu to use. Set it to empty means to use cpu.')


def decode(data_loader):
    """Decode given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    translation_out = []
    all_inst_ids = []
    post_out = []
    for batch_id, (src_seq, src_valid_length, inst_ids) in enumerate(data_loader):

        xpu_src_seq = utils.split_and_load(src_seq, ctx)
        xpu_src_valid_length = utils.split_and_load(src_valid_length, ctx)

        # Calculating Loss
        for xpu_X, xpu_XL in zip(xpu_src_seq, xpu_src_valid_length):
            out = model(xpu_X)
            posterior = mx.ndarray.softmax(out, axis=2)
            # Greedy search translate
            trans = mx.ndarray.argmax(out, axis=2).asnumpy() # shape (batch_size, sequence_length, vocab_size)
            for i in range(trans.shape[0]):
                input_len = xpu_XL[i].astype(np.dtype('int')).asscalar()
                tmp_tans = []
                for j in range(input_len):
                    prev = int(trans[i][j-1])
                    curr = int(trans[i][j])
                    if j == 0 or curr != prev:
                        if curr != 0: # means is blank
                            tmp_tans.append(curr)
                translation_out.append(
                    [ tgt_vocab.idx_to_token[ele] for ele in tmp_tans ] )
                post_out.append(posterior[i].asnumpy())

        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        logger.info('[Batch {}/{}]'.format(batch_id+1, len(data_loader)))

    real_translation_out = [None for _ in range(len(all_inst_ids))]
    real_post_out = [None for _ in range(len(all_inst_ids))]
    for ind, sentence, post in zip(all_inst_ids, translation_out, post_out):
        real_translation_out[ind] = sentence
        real_post_out[ind] = post

    return real_translation_out, real_post_out


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
if args.src_sym is not None:
    src_vocab = Vocab(vocab_file=args.src_sym, unknown_token=None, padding_token=None,
                                              bos_token=None, eos_token=None)
# CTC don't need <s> </s>
if args.tgt_sym is not None:
    tgt_vocab = Vocab(vocab_file=args.tgt_sym, unknown_token=None, padding_token=None,
                                              bos_token=None, eos_token=None)
logger.info(tgt_vocab)


logger.info("=============================")
logger.info("Loding the test data...")
logger.info("=============================")

logger.info("Test set:\n source:{}\n ".format(args.test_src_rspecifier))
data_test = TestReader(args.test_src_rspecifier, args.apply_exp)

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

# for 1-of-k or posterior input
src_embed = gluon.nn.HybridSequential(prefix='src_embed_')
with src_embed.name_scope():
    src_embed.add(gluon.nn.Dense(units=args.hidden_size, weight_initializer=mx.init.Uniform(0.1), flatten=False))
    src_embed.add(gluon.nn.Dropout(rate=args.embed_dropout))

model = Nnet(src_embed=src_embed,
            units=len(tgt_vocab),
            hidden_size=args.hidden_size, 
            cell_type=args.cell, 
            num_layers=args.num_layers, 
            cell_dropout=args.cell_dropout)

logger.info(model.collect_params().keys())

if args.init_model == '':
    logger.error("no model specified !")
else:
    model.load_parameters(args.init_model, ctx=ctx)
# model.hybridize()

logger.info(model)

test_translation_out, test_post_out = decode(test_data_loader)

assert len(test_translation_out) == len(data_test), 'the # of translation shoule be same as the test set'
assert len(test_post_out) == len(data_test), 'the # of posterior matrix shoule be same as the test set'

trans_file = os.path.join(decode_dir, 'result.txt')
with io.open(trans_file, 'w', encoding='utf-8') as of:
    for i in range(len(data_test)):
        sent = ' '.join(test_translation_out[i])
        key = data_test.get_utt_key(i)
        of.write(key + ' ' + sent + '\n')
        
post_file = os.path.join(decode_dir, 'post.ark')
with io.open(post_file, 'wb') as of:
    for i in range(len(data_test)):
        key = data_test.get_utt_key(i)
        mat = test_post_out[i]
        kaldi_io.write_mat(of, mat, key)
    