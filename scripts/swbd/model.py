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
"""Machine translation models and translators."""


__all__ = ['NMTModel', 'BeamSearchTranslator']

import warnings
import numpy as np
from mxnet.gluon import HybridBlock, Block
from mxnet.gluon import nn, rnn
import mxnet as mx
from gluonE2EASR.model import BeamSearchScorer, BeamSearchSampler

import sys

class NMTModel(Block):
    """Model for Neural Machine Translation.

    Parameters
    ----------
    src_vocab : Vocab
        Source vocabulary.
    tgt_vocab : Vocab
        Target vocabulary.
    encoder : Seq2SeqEncoder
        Encoder that encodes the input sentence.
    decoder : Seq2SeqDecoder
        Decoder that generates the predictions based on the output of the encoder.
    embed_size : int or None, default None
        Size of the embedding vectors. It is used to generate the source and target embeddings
        if src_embed and tgt_embed are None.
    embed_dropout : float, default 0.0
        Dropout rate of the embedding weights. It is used to generate the source and target
        embeddings if src_embed and tgt_embed are None.
    embed_initializer : Initializer, default mx.init.Uniform(0.1)
        Initializer of the embedding weights. It is used to generate ghe source and target
        embeddings if src_embed and tgt_embed are None.
    src_embed : Block or None, default None
        The source embedding. If set to None, src_embed will be constructed using embed_size and
        embed_dropout.
    tgt_embed : Block or None, default None
        The target embedding. If set to None and the tgt_embed will be constructed using
        embed_size and embed_dropout. Also if `share_embed` is turned on, we will set tgt_embed
        to be the same as src_embed.
    share_embed : bool, default False
        Whether to share the src/tgt embeddings or not.
    tgt_proj : Block or None, default None
        Layer that projects the decoder outputs to the target vocabulary.
    prefix : str or None
        See document of `Block`.
    params : ParameterDict or None
        See document of `Block`.
    """
    def __init__(self, src_vocab, tgt_vocab, encoder, decoder,
                 embed_size=None, embed_dropout=0.0, embed_initializer=mx.init.Uniform(0.1),
                 src_embed=None, tgt_embed=None, share_embed=False, tgt_proj=None,
                 prefix=None, params=None):
        super(NMTModel, self).__init__(prefix=prefix, params=params)
        self.tgt_vocab = tgt_vocab
        self.src_vocab = src_vocab
        self.encoder = encoder
        self.decoder = decoder
        self._shared_embed = share_embed
        if embed_dropout is None:
            embed_dropout = 0.0
        # Construct src embedding
        if share_embed and tgt_embed is not None:
            warnings.warn('"share_embed" is turned on and \"tgt_embed\" is not None. '
                          'In this case, the provided "tgt_embed" will be overwritten by the '
                          '"src_embed". Is this intended?')
        if src_embed is None:
            assert embed_size is not None, '"embed_size" cannot be None if "src_embed" is not ' \
                                           'given.'
            with self.name_scope():
                self.src_embed = nn.HybridSequential(prefix='src_embed_')
                with self.src_embed.name_scope():
                    self.src_embed.add(nn.Embedding(input_dim=len(src_vocab), output_dim=embed_size,
                                                    weight_initializer=embed_initializer))
                    self.src_embed.add(nn.Dropout(rate=embed_dropout))
        else:
            self.src_embed = src_embed
        # Construct tgt embedding
        if share_embed:
            self.tgt_embed = self.src_embed
        else:
            if tgt_embed is not None:
                self.tgt_embed = tgt_embed
            else:
                assert embed_size is not None,\
                    '"embed_size" cannot be None if "tgt_embed" is ' \
                    'not given and "shared_embed" is not turned on.'
                with self.name_scope():
                    self.tgt_embed = nn.HybridSequential(prefix='tgt_embed_')
                    with self.tgt_embed.name_scope():
                        self.tgt_embed.add(
                            nn.Embedding(input_dim=len(tgt_vocab), output_dim=embed_size,
                                         weight_initializer=embed_initializer))
                        self.tgt_embed.add(nn.Dropout(rate=embed_dropout))
        # Construct tgt proj
        if tgt_proj is None:
            with self.name_scope():
                self.tgt_proj = nn.Dense(units=len(tgt_vocab), flatten=False, prefix='tgt_proj_')
        else:
            self.tgt_proj = tgt_proj

    def encode(self, inputs, valid_length=None):
        """Encode the input sequence.

        Parameters
        ----------
        inputs : NDArray
        valid_length : NDArray or None, default None

        Returns
        -------
        outputs : NDArrays or Symbols
            The output data. Shape (batch_size, sequence_length, units)
        last_states : list of NDArrays/Symbols or NDArrays/Symbols
            The last valid rnn state in the sequence. 
            For lstm, it should be a list of two initial recurrent state tensors [c, h]
            Each has shape (num_layers, batch_size, num_hidden)

            Notes: since we forward the padded sequence at one time, the last_sate
                   will contain the padding forward results
        """
        return self.encoder(self.src_embed(inputs), valid_length)

    def decode_seq(self, inputs, decoder_states, valid_length=None):
        """Decode given the input sequence.

        Parameters
        ----------
        inputs : NDArray. Shape (batch_size, )
        decoder_states : list of NDArrays
            The decoder states, includes:
                - rnn_states : list of NDArrays/Symbols or NDArrays/Symbols
                - mem_value : NDArray
                - mem_masks : NDArray, optional
        valid_length : NDArray or None, default None

        Returns
        -------
        outputs : NDArray
            The outputs of the decoder. Shape is (batch_size, tgt_src_len, units)
        attention_weights : NDArray
            The attention weights of the decoder. 
            Shape is (batch_size, src_seq_length, tgt_src_len)
        """
        rnn_outputs, attention_vecs, attention_weights =\
            self.decoder.decode_seq(inputs=self.tgt_embed(inputs),
                                    decoder_states=decoder_states,
                                    valid_length=valid_length)

        combine_outpus = mx.nd.concat(rnn_outputs, attention_vecs, dim=2)
        outputs = self.tgt_proj(combine_outpus)
        return outputs, attention_weights

    def decode_step(self, step_input, decoder_states):
        """One step decoding of the translation model.

        Parameters
        ----------
        step_input : NDArray
            Shape (batch_size,)
        decoder_states : list of NDArrays -> [rnn_state, mem_value, mem_masks]

        Returns
        -------
        step_output : NDArray
            Shape (batch_size, units)
        decoder_states : list
        attention_weight : NDArray
            Additional outputs of the step, e.g, the attention weights
        """
        step_rnn_output, new_decoder_states, context_vec, attention_weight =\
            self.decoder(self.tgt_embed(step_input), decoder_states)

        combine_step_output = mx.nd.concat(step_rnn_output, context_vec, dim=1)
        step_output = self.tgt_proj(combine_step_output)
        return step_output, new_decoder_states, attention_weight

    def __call__(self, src_seq, tgt_seq, src_valid_length=None, tgt_valid_length=None):  #pylint: disable=arguments-differ
        """Generate the prediction given the src_seq and tgt_seq.

        This is used in training an NMT model.

        Parameters
        ----------
        src_seq : NDArray
        tgt_seq : NDArray
        src_valid_length : NDArray or None
        tgt_valid_length : NDArray or None

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, tgt_length, tgt_word_num)
        attention_weight : NDArray
        """
        return super(NMTModel, self).__call__(src_seq, tgt_seq, src_valid_length, tgt_valid_length)

    def forward(self, src_seq, tgt_seq, src_valid_length=None, tgt_valid_length=None):  #pylint: disable=arguments-differ
        encode_mems, enc_rnn_states = self.encode(src_seq, valid_length=src_valid_length)
        decoder_states = self.decoder.init_state_from_encoder(encode_mems,
                                                              enc_rnn_states,
                                                              encoder_valid_length=src_valid_length)

        return self.decode_seq(tgt_seq, decoder_states, tgt_valid_length)

def _nested_swap_states(states):
    ''' swap the 0 , 1 axis of the state

    Parameters
    ----------
    states : list of NDArrays/Symbols or NDArrays/Symbols
        The last valid encoder rnn state in the sequence. 
        For lstm, it should be a list of two initial recurrent state tensors [c, h]
        Each has shape (num_layers, batch_size, num_hidden) 
                    or (batch_size, num_layers, num_hidden)

        Notes: since we forward the padded sequence at one time, the last_sate
               will contain the padding forward results 

    Returns 
    -------
    out_states :  list of NDArrays/Symbols or NDArrays/Symbols
        Each element has the shape (batch_size, num_layers, num_hidden)
                                or (num_layers, batch_size, num_hidden) 
    '''
    if isinstance(states, (mx.sym.Symbol, mx.nd.NDArray)):
        F = mx.sym if isinstance(states, mx.sym.Symbol) else mx.ndarray
        return F.swapaxes(states, 0, 1)
    elif isinstance(states, list):
        ret = []
        for ele in states:
            ret.append(_nested_swap_states(ele))
        return ret
    else:
        raise NotImplementedError

class BeamSearchTranslator(object):
    """Beam Search Translator

    Parameters
    ----------
    model : NMTModel
        The neural machine translation model
    beam_size : int
        Size of the beam
    scorer : BeamSearchScorer
        Score function used in beamsearch
    max_length : int
        The maximum decoding length
    """
    def __init__(self, model, beam_size=1, scorer=BeamSearchScorer(), max_length=100):
        self._model = model
        self._sampler = BeamSearchSampler(
            decoder=self._decode_logprob,
            beam_size=beam_size,
            eos_id=model.tgt_vocab.token_to_idx[model.tgt_vocab.eos_token],
            scorer=scorer,
            max_length=max_length)

    def _decode_logprob(self, step_input, decoder_states):
        """One step decoding of the translation model.

        Parameters
        ----------
        step_input : NDArray
            Shape (batch_size, C)
        decoder_states : list of NDArrays -> [rnn_state(B, L, C), mem_value, mem_masks]

        Returns
        -------
        step_output : NDArray
            Shape (batch_size, units)
        decoder_states : list
        attention_weight : NDArray
            Additional outputs of the step, e.g, the attention weights
            
        Notes:  The states used by Beamsearch should have shape (batch_size, ...)
                But the rnn cells always keep the states as shape (num_layers, batch_size, hidden)
                Hence, we need to swap the axes of the rnn_states every time
        """
        batch_size = step_input.shape[0]
        assert isinstance(decoder_states, list)
        assert batch_size == decoder_states[0][0].shape[0] # the first axis should be batch_size

        # change the rnn states to shape (L, B, C)
        decoder_states[0] = _nested_swap_states(decoder_states[0])
        out, new_decoder_states, _ = self._model.decode_step(step_input, decoder_states)
        # rechange the rnn state to shape (B, L, C)
        new_decoder_states[0] = _nested_swap_states(new_decoder_states[0])

        return mx.nd.log_softmax(out), new_decoder_states

    def translate(self, src_seq, src_valid_length):
        """Get the translation result given the input sentence.

        Parameters
        ----------
        src_seq : mx.nd.NDArray
            Shape (batch_size, length)
        src_valid_length : mx.nd.NDArray
            Shape (batch_size,)

        Returns
        -------
        samples : NDArray
            Samples draw by beam search. Shape (batch_size, beam_size, length). dtype is int32.
        scores : NDArray
            Scores of the samples. Shape (batch_size, beam_size). We make sure that scores[i, :] are
            in descending order.
        valid_length : NDArray
            The valid length of the samples. Shape (batch_size, beam_size). dtype will be int32.
        """
        batch_size = src_seq.shape[0]
        encode_mems, enc_rnn_states = self._model.encode(src_seq, valid_length=src_valid_length)
        begin_decoder_states = self._model.decoder.init_state_from_encoder(encode_mems,
                                                                           enc_rnn_states,
                                                                           src_valid_length)
        # change the rnn states to shape (B, L, C)
        begin_decoder_states[0] = _nested_swap_states(begin_decoder_states[0])

        inputs = mx.nd.full(shape=(batch_size,), ctx=src_seq.context, dtype=np.float32,
                            val=self._model.tgt_vocab.token_to_idx[self._model.tgt_vocab.bos_token])
        samples, scores, sample_valid_length = self._sampler(inputs, begin_decoder_states)
        return samples, scores, sample_valid_length

class Nnet(HybridBlock):
    """Model for Neural Machine Translation.

    Parameters
    ----------
    units : int
        Dimensionality of the output space.
    hidden_size: int, default 512
        The number of features in the hidden state h.
    cell_type : str, [ lstm | bilstm | gru ]
        The rnn cell type
    num_layers : int, default 3
        Number of recurrent layers.      
    embed_dropout : float, default 0.0
        Dropout rate of the embedding weights.
    cell_dropout : float, default 0.0
        Dropout rate of the RNN cells.
    in_units : int, optional
        Size of the input data. If not specified, initialization will be
        deferred to the first time `forward` is called and `in_units`
        will be inferred from the shape of input data.
    prefix : str or None
        See document of `Block`.
    params : ParameterDict or None
        See document of `Block`.
    """
    def __init__(self, units, hidden_size=512, cell_type='lstm', num_layers=3, embed_dropout=0.0, 
                    cell_dropout=0.0, in_units=0, src_embed=None, tgt_proj=None, prefix=None, **kwargs):
        super(Nnet, self).__init__(prefix=prefix, **kwargs)

        # Construct src embedding
        if src_embed is None:
            assert in_units != 0, '"in_units" cannot be None if "src_embed" is not ' \
                                           'given.'
            with self.name_scope():
                self.src_embed = nn.Sequential(prefix='src_embed_')
                with self.src_embed.name_scope():
                    self.src_embed.add(nn.Embedding(input_dim=in_units, output_dim=hidden_size,
                                                weight_initializer=mx.init.Uniform(0.1), flatten=False))
                    self.src_embed.add(nn.Dropout(rate=embed_dropout))
        else:
            self.src_embed = src_embed

        # Construct tgt proj
        if tgt_proj is None:
            with self.name_scope():
                self.tgt_proj = nn.Dense(units=units, flatten=False, prefix='tgt_proj_')
        else:
            self.tgt_proj = tgt_proj

        # Construct RNN layes
        if cell_type == 'lstm':
            with self.name_scope():
                self.net = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, layout='NTC', 
                                    dropout=cell_dropout, bidirectional=False)
        elif cell_type == 'bilstm':
            with self.name_scope():
                self.net = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, layout='NTC',
                                    dropout=cell_dropout, bidirectional=True)
        elif cell_type == 'gru':
            with self.name_scope():
                self.net = rnn.GRU(hidden_size=hidden_size, num_layers=num_layers, layout='NTC',
                                    dropout=cell_dropout, bidirectional=False)
        elif cell_type == 'bigru':
            with self.name_scope():
                self.net = rnn.GRU(hidden_size=hidden_size, num_layers=num_layers, layout='NTC',
                                    dropout=cell_dropout, bidirectional=True)
        else:
            print('Not supported layer type: {}'.format(cell_type))
            sys.exit(0)

    def hybrid_forward(self, F, inputs):
        embed = self.src_embed(inputs)
        net_out, _ = self.net(embed)
        return self.tgt_proj(net_out)
