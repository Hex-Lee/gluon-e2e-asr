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
from gluonE2EASR.log import setup_main_logger, log_mxnet_version
from gluonE2EASR.vocab import Vocab

from model import Nnet
from loss import CtcLoss, SoftmaxCEMaskedLoss
from wer import compute_wer

from reader_kaldi_io import Reader
import utils

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

############################################
########## Arguments specify ###############
############################################
parser = argparse.ArgumentParser(description='SwithBorad 300h End-to-End ASR Example.'
                                             'We train the Google NMT model')
# parser.add_mutually_exclusive_group(required=False)

parser.add_argument('--src_rspecifier', type=str, help='training feature specifier(with kaldi scp/ark format)')
parser.add_argument('--tgt_rspecifier', type=str, help='training target sentences specifier(with kaldi scp/ark format)')
parser.add_argument('--cv_src_rspecifier', type=str, help='valid data feature specifier(with kaldi scp/ark format)')
parser.add_argument('--cv_tgt_rspecifier', type=str, help='valid target sentences specifier(with kaldi scp/ark format)')

parser.add_argument('--src_sym', type=str, default=None, help='[optional] source vocab')
parser.add_argument('--tgt_sym', type=str, default=None, help='target vocab.')

parser.add_argument('--init_model', type=str, default='', help='the initail model other than randon initailize')
parser.add_argument('--num_layers', type=int, default=3, help='number of layers in the encoder')
parser.add_argument('--cell', type=str, default='lstm', help='the cell type [ lstm | bilstm | gru ]')
parser.add_argument('--hidden_size', type=int, default=300, help='Dimension of the embedding vectors and states.')
parser.add_argument('--embed_dropout', type=float, default=0, help='dropout applied to embedding layers (0 = no dropout)')
parser.add_argument('--cell_dropout', type=float, default=0, help='dropout applied to RNN cell layers (0 = no dropout)')

parser.add_argument('--num_gpus', type=int, default=None, help='number of the gpu to use. Set it to empty means to use cpu.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_buckets', type=int, default=5, help='Bucket number')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the throughput of the bucketing')
parser.add_argument('--train_shuffle', default=False, action='store_true', help='shuffle the train data')
parser.add_argument('--ascending', dest='descending', action='store_false', help='Train the data with ascending order'
                                                                                    '(useless when setting shuffle)')
parser.add_argument('--descending', dest='descending', action='store_true', help='Train the data with descending order'
                                                                                    '(useless when setting shuffle)')
parser.set_defaults(descending=True)

parser.add_argument('--loss_function', type=str, default='ctc', help='loss function [ctc|ce]')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=1E-3, help='Initial learning rate')
parser.add_argument('--lr_update_factor', type=float, default=0.5, help='Learning rate decay factor')
parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
parser.add_argument('--use_grad_clip', default=False, action='store_true', help='use gradient clipping')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')

parser.add_argument('--max_epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--halving_after_epoch', type=int, default=1, help='halving lr becomes enabled after this many epochs') 
parser.add_argument('--terminal_after_epoch', type=int, default=3, help='early terminal when the cv loss do not '
                                                                        'descrease after this many epochs') 

parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--save_dir', type=str, default='exp_debug', help='directory path to save the final model and training log')

parser.add_argument('--lr_searching', default=False, action='store_true', help='searching the learning_rate')


def evaluate(data_loader):
    """Evaluate given the data loader

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
    avg_loss = 0.0
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
            in enumerate(data_loader):

        xpu_src_seq = utils.split_and_load(src_seq, ctx)
        xpu_tgt_seq = utils.split_and_load(tgt_seq, ctx)
        xpu_src_valid_length = utils.split_and_load(src_valid_length, ctx)
        xpu_tgt_valid_length = utils.split_and_load(tgt_valid_length, ctx)

        # Calculating Loss
        for xpu_X, xpu_y, xpu_XL, xpu_yl in zip(xpu_src_seq, xpu_tgt_seq,
                                                          xpu_src_valid_length, xpu_tgt_valid_length):
            out = model(xpu_X)

            if args.loss_function == 'ctc':
                loss = loss_function(out, xpu_y, xpu_XL, xpu_yl).mean().asscalar()
            elif args.loss_function == 'ce':
                loss = loss_function(out, xpu_y, xpu_XL).mean().asscalar()
            avg_loss += loss

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

        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())

    real_translation_out = [None for _ in range(len(all_inst_ids))]
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = sentence

    return avg_loss / ( len(ctx)*len(data_loader) ), real_translation_out

def write_sentences(sentences, file_path):
    with io.open(file_path, 'w', encoding='utf-8') as of:
        for sent in sentences:
            of.write(' '.join(sent) + '\n')


############################################
########## Arguments handle  ###############
############################################
args = parser.parse_args()
if args.save_dir is None:
    args.save_dir = os.path.join(os.getcwd(), name)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

model_dir = os.path.join(args.save_dir, 'params')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
valid_out_dir = os.path.join(args.save_dir, 'valid_out')
if not os.path.exists(valid_out_dir):
    os.makedirs(valid_out_dir)

log_file = os.path.join(args.save_dir, 'train.log')
if os.path.exists(log_file):
    os.remove(log_file)

logger = setup_main_logger(name=__name__, path=log_file)
log_mxnet_version(logger)
logger.info("hostname: {}".format(socket.gethostname()))
logger.info(args)

############################################
############### Data loading ###############
############################################
logger.info("==================")
logger.info("Loding the data...")
logger.info("==================")

if args.src_sym is not None:
    src_vocab = Vocab(vocab_file=args.src_sym, unknown_token=None, padding_token=None,
                                              bos_token=None, eos_token=None)
# CTC don't need <s> </s>
if args.tgt_sym is not None:
    tgt_vocab = Vocab(vocab_file=args.tgt_sym, unknown_token=None, padding_token=None,
                                              bos_token=None, eos_token=None)
logger.info(tgt_vocab)

logger.info("Train set:\n source:{}\n target:{}".format(args.src_rspecifier, args.tgt_rspecifier))
data_train = Reader(args.src_rspecifier, args.tgt_rspecifier)

logger.info("Valid set:\n source:{}\n target:{}".format(args.cv_src_rspecifier, args.cv_tgt_rspecifier))
data_val = Reader(args.cv_src_rspecifier, args.cv_tgt_rspecifier)

data_train_lengths = data_train.get_valid_length()
data_val_lengths = data_val.get_valid_length()

val_tgt_sentences = []
for _, tgt_sentence, _, _, _, in data_val:
    tmp = [tgt_vocab.idx_to_token[int(ele)] for ele in tgt_sentence]
    val_tgt_sentences.append(tmp)

with io.open(os.path.join(valid_out_dir, 'val_gt.txt'), 'w') as of:
    for ele in val_tgt_sentences:
        of.write(' '.join(ele) + '\n')

batchify_fn = btf.Tuple(btf.Pad(axis=0), btf.Pad(pad_val=0), btf.Stack(), btf.Stack(), btf.Stack())

train_batch_sampler = FixedBucketSampler(lengths=data_train_lengths,
                                         batch_size=args.batch_size,
                                         num_buckets=args.num_buckets,
                                         ratio=args.bucket_ratio,
                                         shuffle=args.train_shuffle,
                                         reverse=args.descending)
logger.info('Train Batch Sampler:\n{}'.format(train_batch_sampler.stats()))
train_data_loader = DataLoader(data_train,
                               batch_sampler=train_batch_sampler,
                               batchify_fn=batchify_fn,
                               num_workers=1)

val_batch_sampler =  FixedBucketSampler(lengths=data_val_lengths,
                                       batch_size=args.batch_size,
                                       num_buckets=args.num_buckets,
                                       ratio=args.bucket_ratio,
                                       shuffle=False)
logger.info('Valid Batch Sampler:\n{}'.format(val_batch_sampler.stats()))
val_data_loader = DataLoader(data_val,
                             batch_sampler=val_batch_sampler,
                             batchify_fn=batchify_fn,
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

if args.init_model == '':
    model.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
else:
    model.load_parameters(args.init_model, ctx=ctx)
model.hybridize()

if args.loss_function == 'ctc':
    loss_function = CtcLoss(layout='NTC', label_layout='NT')
elif args.loss_function == 'ce':
    loss_function = SoftmaxCEMaskedLoss()
else:
    logger.error("Undefined loss function!")
loss_function.hybridize()

############################################
########## Initial Valid forward ###########
############################################
valid_loss, valid_translation_out = evaluate(val_data_loader)
valid_wer = compute_wer(val_tgt_sentences, valid_translation_out)
# Since the gluon has delay initialize, the model will be initialed when first batch feed in
# hence we can get the inferred model shape
logger.info(model)
logger.info('[Init] valid Loss={:.4f}, valid wer={:.2f}'
             .format(valid_loss, valid_wer * 100))
write_sentences(valid_translation_out,
                os.path.join(valid_out_dir, 'epoch0_valid_out.txt'))

model_name = 'param.{:02d}_cv{:.2f}'.format(0, valid_loss)
save_path = os.path.join(model_dir, model_name)

model.save_parameters(save_path)
logger.info('Save the initial model parameters to {}'.format(save_path))
link_path = os.path.join(args.save_dir, 'valid_best.params')
utils.symlink_force('params/'+model_name, link_path)
logger.info('Link best parameters {{ {} }} to {{ {} }}'.format(save_path, link_path))


############################################
############# Training process #############
############################################
def train():
    """Training function."""
    if args.optimizer == 'sgd':
        train_opts = {'learning_rate': args.lr, 'momentum': args.momentum}
    else:
        train_opts = {'learning_rate': args.lr}

    trainer = gluon.Trainer(model.collect_params(), args.optimizer, train_opts)

    logger.info("optimizer: {}".format(args.optimizer))
    logger.info("==================")
    logger.info("Start training...")
    logger.info("==================")

    best_valid_loss = 1e10
    terminal_flag = 0
    for epoch_id in range(args.max_epochs):
        log_avg_seq_len = 0
        log_avg_loss = 0
        log_avg_gnorm = 0
        log_wc = 0
        epoch_train_loss = 0
        epoch_start_time = time.time()
        log_start_time = time.time()
        for batch_id, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, _)\
                in enumerate(train_data_loader):
            # logger.info(src_seq.context) Context suddenly becomes GPU.

            xpu_src_seq = utils.split_and_load(src_seq, ctx)
            xpu_tgt_seq = utils.split_and_load(tgt_seq, ctx)
            xpu_src_valid_length = utils.split_and_load(src_valid_length, ctx)
            xpu_tgt_valid_length = utils.split_and_load(tgt_valid_length, ctx)

            with mx.autograd.record():
                for xpu_X, xpu_y, xpu_XL, xpu_yl in zip(xpu_src_seq, xpu_tgt_seq,
                                                          xpu_src_valid_length, xpu_tgt_valid_length):

                    out = model(xpu_X) # remove <eos>
                    if args.loss_function == 'ctc':
                        loss = loss_function(out, xpu_y, xpu_XL, xpu_yl).mean()
                    elif args.loss_function == 'ce':
                        loss = loss_function(out, xpu_y, xpu_XL).mean()
                    loss.backward()
                    log_avg_loss += loss.asscalar()
                    epoch_train_loss += loss.asscalar()

            gnorm = 0.0
            if args.use_grad_clip:
                for context in ctx:
                    grads = [p.grad(context) for p in model.collect_params().values()]
                    gnorm += gluon.utils.clip_global_norm(grads, args.clip)

            trainer.step(1)

            src_wc = src_valid_length.sum().asscalar()
            tgt_wc = tgt_valid_length.sum().asscalar()              

            log_avg_seq_len += src_seq.shape[1]
            log_avg_gnorm += gnorm
            log_wc += src_wc + tgt_wc
            if (batch_id + 1) % args.log_interval == 0:
                wps = log_wc / (time.time() - log_start_time)
                logger.info('[Epoch {} Batch {}/{}] loss={:.4f}, gnorm={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K, sequence-length={:.2f}'
                             .format(epoch_id + 1, batch_id + 1, len(train_data_loader),
                                     log_avg_loss / ( args.log_interval * len(ctx) ),
                                     log_avg_gnorm / ( args.log_interval * len(ctx) ),
                                     wps / 1000, log_wc / 1000,
                                     log_avg_seq_len / args.log_interval))
                log_start_time = time.time()
                log_avg_seq_len = 0
                log_avg_loss = 0
                log_avg_gnorm = 0
                log_wc = 0

                ### For learning rate search ###
                if args.lr_searching:
                    new_lr = trainer.learning_rate * args.lr_update_factor
                    trainer.set_learning_rate(new_lr)
                    logger.info('Learning rate change to {}, terminal_flag={}'.format(new_lr, terminal_flag))
        ######################## End of mini-batch training #############################
        epoch_train_loss = epoch_train_loss / ( len(train_data_loader) * len(ctx) )
        valid_loss, valid_translation_out = evaluate(val_data_loader)
        valid_wer = compute_wer(val_tgt_sentences, valid_translation_out)
        logger.info('[Epoch {}] time cost={:.1f}, lr={}, valid Loss={:.4f}, valid wer={:.2f}'
                     .format(epoch_id + 1, time.time()-epoch_start_time, 
                                trainer.learning_rate, valid_loss, valid_wer * 100))
        write_sentences(valid_translation_out,
                        os.path.join(valid_out_dir, 'epoch{:d}_valid_out.txt').format(epoch_id + 1))
        
        model_name = 'param.{:02d}.tr{:.2f}_cv{:.2f}'.format(epoch_id + 1, epoch_train_loss, valid_loss)
        save_path = os.path.join(model_dir, model_name)
        model.save_parameters(save_path)
        logger.info('Save the Epoch {} parameters to {}'.format(epoch_id + 1, save_path))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            utils.symlink_force('params/'+model_name, link_path)
            logger.info('Link best parameters {{ {} }} to {{ {} }}'.format(save_path, link_path))

            terminal_flag = 0
        else:
            if (epoch_id+1 > args.halving_after_epoch):
                new_lr = trainer.learning_rate * args.lr_update_factor
                trainer.set_learning_rate(new_lr)
                logger.info('Learning rate change to {}, terminal_flag={}'.format(new_lr, terminal_flag))
                terminal_flag += 1
                # reset the model to the best model
                model.load_parameters(link_path)
                logger.info('Reload the best params: {}'.format(link_path))

        if (terminal_flag >= args.terminal_after_epoch):
            break
    ######################## End of the Epoch training #############################
    logger.info("==================")
    logger.info("Success Trained!")
    logger.info("==================")

if __name__ == '__main__':
    train()
