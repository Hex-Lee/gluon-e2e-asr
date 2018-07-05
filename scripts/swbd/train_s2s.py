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

from encoder_decoder import get_nmt_encoder_decoder
from model import NMTModel, BeamSearchTranslator
from loss import SoftmaxCEMaskedLoss
from bleu import compute_bleu
from wer import compute_wer

from reader_kaldi_io import Reader
import utils
import _constants as _C

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

############################################
########## Arguments specify ###############
############################################
parser = argparse.ArgumentParser(description='SwithBorad 300h End-to-End ASR Example.'
                                             'We train the Google NMT model')
parser.add_argument('--src_rspecifier', type=str, help='training feature specifier(with kaldi scp/ark format)')
parser.add_argument('--tgt_rspecifier', type=str, help='training target sentences specifier(with kaldi scp/ark format)')
parser.add_argument('--src_sym', type=str, default=None, help='[optional] source vocab')
parser.add_argument('--tgt_sym', type=str, default=None, help='target vocab.'
                                    'reserved words:'
                                    '\'<bos>\': begin of sentance; \'<eos>\': end of sentance;'
                                    '\'<unk>\': unknown token; \'<pad>\': padding token')

parser.add_argument('--cv_src_rspecifier', type=str, help='valid data feature specifier(with kaldi scp/ark format)')
parser.add_argument('--cv_tgt_rspecifier', type=str, help='valid target sentences specifier(with kaldi scp/ark format)')

parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--hidden_size', type=int, default=300, help='Dimension of the embedding '
                                                                'vectors and states.')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--residual', type=bool, default=False,
                    help='use residual in encoder and decoder or not')

parser.add_argument('--num_enc_layers', type=int, default=3, help='number of layers in the encoder')
parser.add_argument('--num_enc_bi_layers', type=int, default=0,
                    help='number of bidirectional layers in the encoder')
parser.add_argument('--num_dec_layers', type=int, default=3, help='number of layers in the decoder')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--beam_size', type=int, default=1, help='Beam size')
parser.add_argument('--lp_alpha', type=float, default=1.0,
                    help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')
parser.add_argument('--num_buckets', type=int, default=5, help='Bucket number')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the '
                                                                    'throughput of the bucketing')
parser.add_argument('--src_max_len', type=int, default=100, help='Maximum length of the source '
                                                                'sentence')
parser.add_argument('--tgt_max_len', type=int, default=100, help='Maximum length of the target '
                                                                'sentence')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=1E-3, help='Initial learning rate')
parser.add_argument('--lr_update_factor', type=float, default=0.5,
                    help='Learning rate decay factor')

parser.add_argument('--use_grad_clip', type=bool, default=False, help='whether to use gradient clipping')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')

parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='exp_debug',
                    help='directory path to save the final model and training log')
parser.add_argument('--num_gpus', type=int, default=None,
                    help='number of the gpu to use. Set it to empty means to use cpu.')


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
    avg_loss_denom = 0
    avg_loss = 0.0
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
            in enumerate(data_loader):

        xpu_src_seq = utils.split_and_load(src_seq, ctx)
        xpu_tgt_seq = utils.split_and_load(tgt_seq, ctx)
        xpu_src_valid_length = utils.split_and_load(src_valid_length, ctx)
        xpu_tgt_valid_length = utils.split_and_load(tgt_valid_length, ctx)

        # Calculating Loss
        batch_loss = []
        for xpu_X, xpu_y, xpu_XL, xpu_yl in zip(xpu_src_seq, xpu_tgt_seq,
                                                          xpu_src_valid_length, xpu_tgt_valid_length):
            out, _ = model(xpu_X, xpu_y[:, :-1], xpu_XL, xpu_yl - 1) # remove <eos>
            loss = loss_function(out, xpu_y[:, 1:], xpu_yl - 1).mean().asscalar() # remove <bos>

            avg_loss += loss * (xpu_y.shape[1] - 1)
            avg_loss_denom += (xpu_y.shape[1] - 1)

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

        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())

    real_translation_out = [None for _ in range(len(all_inst_ids))]
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = sentence
    # Average loss
    avg_loss = avg_loss / avg_loss_denom

    return avg_loss, real_translation_out


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
    src_vocab = Vocab(vocab_file=args.src_sym, unknown_token=None, padding_token='<blk>',
                                              bos_token=None, eos_token=None)
if args.tgt_sym is not None:
    tgt_vocab = Vocab(vocab_file=args.tgt_sym, padding_token='<blk>')

logger.info("Train set:\n source:{}\n target:{}".format(args.src_rspecifier, args.tgt_rspecifier))
data_train = Reader(args.src_rspecifier, args.tgt_rspecifier,
                        tgt_vocab[tgt_vocab.bos_token], tgt_vocab[tgt_vocab.eos_token])

logger.info("Valid set:\n source:{}\n target:{}".format(args.cv_src_rspecifier, args.cv_tgt_rspecifier))
data_val = Reader(args.cv_src_rspecifier, args.cv_tgt_rspecifier,
                        tgt_vocab[tgt_vocab.bos_token], tgt_vocab[tgt_vocab.eos_token])

data_train_lengths = data_train.get_valid_length()
data_val_lengths = data_val.get_valid_length()

val_tgt_sentences = []
for _, tgt_sentence, _, _, _, in data_val:
    tmp = [tgt_vocab.idx_to_token[ele] for ele in tgt_sentence[1:-1]]
    val_tgt_sentences.append(tmp)

with io.open(os.path.join(valid_out_dir, 'val_gt.txt'), 'w') as of:
    for ele in val_tgt_sentences:
        of.write(' '.join(ele) + '\n')

batchify_fn = btf.Tuple(btf.Pad(axis=0), btf.Pad(), btf.Stack(), btf.Stack(), btf.Stack())
train_batch_sampler = FixedBucketSampler(lengths=data_train_lengths,
                                         batch_size=args.batch_size,
                                         num_buckets=args.num_buckets,
                                         ratio=args.bucket_ratio,
                                         shuffle=True)
logger.info('Train Batch Sampler:\n{}'.format(train_batch_sampler.stats()))
train_data_loader = DataLoader(data_train,
                               batch_sampler=train_batch_sampler,
                               batchify_fn=batchify_fn,
                               num_workers=8)

val_batch_sampler =  FixedBucketSampler(lengths=data_val_lengths,
                                       batch_size=args.batch_size,
                                       num_buckets=args.num_buckets,
                                       ratio=args.bucket_ratio,
                                       shuffle=False)
logger.info('Valid Batch Sampler:\n{}'.format(val_batch_sampler.stats()))
val_data_loader = DataLoader(data_val,
                             batch_sampler=val_batch_sampler,
                             batchify_fn=batchify_fn,
                             num_workers=8)

if args.num_gpus is None:
    ctx = [mx.cpu()]
else:
    ctx = [mx.gpu(i) for i in range(args.num_gpus)]

logger.info(ctx)

############################################
############ Model construction ############
############################################
encoder, decoder = get_nmt_encoder_decoder(hidden_size=args.hidden_size,
                                            dropout=args.dropout,
                                            num_enc_layers=args.num_enc_layers,
                                            num_enc_bi_layers=args.num_enc_bi_layers,
                                            num_dec_layers=args.num_dec_layers,
                                            use_residual=args.residual)
# for 1-of-k or posterior input
src_embed = gluon.nn.HybridSequential(prefix='src_embed_')
with src_embed.name_scope():
    src_embed.add(gluon.nn.Dense(args.hidden_size, in_units=len(src_vocab),
                                  weight_initializer=mx.init.Uniform(0.1), flatten=False))

model = NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 src_embed=src_embed, embed_size=args.hidden_size, prefix='gnmt_')

model.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
# model.hybridize()

translator = BeamSearchTranslator(model=model, beam_size=args.beam_size,
                                  scorer=BeamSearchScorer(alpha=args.lp_alpha,
                                                          K=args.lp_k),
                                  max_length=args.tgt_max_len)
logger.info('Use beam_size={}, alpha={}, K={}'.format(args.beam_size, args.lp_alpha, args.lp_k))


loss_function = SoftmaxCEMaskedLoss()
# loss_function.hybridize()

############################################
########## Initial Valid forward ###########
############################################
valid_loss, valid_translation_out = evaluate(val_data_loader)
valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out)
valid_wer = compute_wer(val_tgt_sentences, valid_translation_out)
# Since the gluon has delay initialize, the model will be initialed when first batch feed in
# hence we can get the inferred model shape
logger.info(model)
logger.info('[Init] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}, valid wer={:.2f}'
             .format(valid_loss, np.exp(valid_loss), valid_bleu_score * 100, valid_wer * 100))
write_sentences(valid_translation_out,
                os.path.join(valid_out_dir, 'epoch0_valid_out.txt'))

save_path = os.path.join(model_dir, 'param.0')
model.save_params(save_path)
logger.info('Save the initial model parameters to {}'.format(save_path))
link_path = os.path.join(args.save_dir, 'valid_best.params')
utils.symlink_force('params/param.0', link_path)
logger.info('Link best parameters {{ {} }} to {{ {} }}'.format(save_path, link_path))

############################################
############# Training process #############
############################################
def train():
    """Training function."""
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr})
    
    logger.info("==================")
    logger.info("Start training...")
    logger.info("==================")
    best_valid_bleu = 0.0
    for epoch_id in range(args.epochs):
        log_avg_loss = 0
        log_avg_gnorm = 0
        log_wc = 0
        log_start_time = time.time()
        for batch_id, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, _)\
                in enumerate(train_data_loader):
            # logger.info(src_seq.context) Context suddenly becomes GPU.
            xpu_src_seq = utils.split_and_load(src_seq, ctx)
            xpu_tgt_seq = utils.split_and_load(tgt_seq, ctx)
            xpu_src_valid_length = utils.split_and_load(src_valid_length, ctx)
            xpu_tgt_valid_length = utils.split_and_load(tgt_valid_length, ctx)

            batch_loss = []
            with mx.autograd.record():
                for xpu_X, xpu_y, xpu_XL, xpu_yl in zip(xpu_src_seq, xpu_tgt_seq,
                                                          xpu_src_valid_length, xpu_tgt_valid_length):

                    out, _ = model(xpu_X, xpu_y[:, :-1], xpu_XL, xpu_yl - 1) # remove <eos>
                    loss = loss_function(out, xpu_y[:, 1:], xpu_yl - 1).mean() # remove <bos>
                    loss = loss * (xpu_y.shape[1] - 1) / (xpu_yl - 1).mean()
                    loss.backward()
                    batch_loss.append(loss)
                    # batch_loss += loss.asscalar()

            gnorm = 0.0
            if args.use_grad_clip:
                for context in ctx:
                    grads = [p.grad(context) for p in model.collect_params().values()]
                    gnorm += gluon.utils.clip_global_norm(grads, args.clip)

            trainer.step(1)

            src_wc = src_valid_length.sum().asscalar()
            tgt_wc = (tgt_valid_length - 1).sum().asscalar()
            for l in batch_loss:
              log_avg_loss += l.asscalar()
            log_avg_gnorm += gnorm
            log_wc += src_wc + tgt_wc
            if (batch_id + 1) % args.log_interval == 0:
                wps = log_wc / (time.time() - log_start_time)
                logger.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, gnorm={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'
                             .format(epoch_id + 1, batch_id + 1, len(train_data_loader),
                                     log_avg_loss / args.log_interval,
                                     np.exp(log_avg_loss / args.log_interval),
                                     log_avg_gnorm / args.log_interval,
                                     wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_avg_gnorm = 0
                log_wc = 0

        valid_loss, valid_translation_out = evaluate(val_data_loader)
        valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out)
        valid_wer = compute_wer(val_tgt_sentences, valid_translation_out)
        logger.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}, valid wer={:.2f}'
                     .format(epoch_id + 1, valid_loss, np.exp(valid_loss), valid_bleu_score * 100, valid_wer * 100))
        write_sentences(valid_translation_out,
                        os.path.join(valid_out_dir, 'epoch{:d}_valid_out.txt').format(epoch_id + 1))
        save_path = os.path.join(model_dir, 'param.{}'.format(epoch_id + 1))
        logger.info('Save the Epoch {} parameters to {}'.format(epoch_id + 1, save_path))
        model.save_params(save_path)

        if valid_bleu_score > best_valid_bleu:
            best_valid_bleu = valid_bleu_score
            utils.symlink_force('params/param.{}'.format(epoch_id + 1), link_path)
            logger.info('Link best parameters {{ {} }} to {{ {} }}'.format(save_path, link_path))
        else:
            new_lr = trainer.learning_rate * args.lr_update_factor
            # reset the model to the best model
            model.load_params(os.path.join(args.save_dir, 'valid_best.params'))
            logger.info('Learning rate change to {}'.format(new_lr))
            trainer.set_learning_rate(new_lr)
    ######################## End of the Epoch training #############################
    model.load_params(os.path.join(args.save_dir, 'valid_best.params'))
    valid_loss, valid_translation_out = evaluate(val_data_loader)
    valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out)
    valid_wer = compute_wer(val_tgt_sentences, valid_translation_out)
    logger.info('Best model valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}, valid wer={:.2f}'
                 .format(valid_loss, np.exp(valid_loss), valid_bleu_score * 100, valid_wer * 100))
    write_sentences(valid_translation_out,
                    os.path.join(valid_out_dir, 'best_valid_out.txt'))

    logger.info("==================")
    logger.info("Success Trained!")
    logger.info("==================")

if __name__ == '__main__':
    train()
