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
"""Encoder and decoder usded in sequence-to-sequence learning."""
__all__ = ['Seq2SeqEncoder', 'Seq2SeqDecoder',
           'NMTEncoder', 'NMTDecoder', 'get_nmt_encoder_decoder']

from functools import partial
import mxnet as mx
from mxnet.base import _as_list
from mxnet.gluon import nn, rnn
from mxnet.gluon.block import Block, HybridBlock
from .attention_cell import AttentionCell, MLPAttentionCell, DotProductAttentionCell

def _get_cell_type(cell_type):
    """Get the object type of the cell by parsing the input

    Parameters
    ----------
    cell_type : str or type

    Returns
    -------
    cell_constructor: type
        The constructor of the RNNCell
    """
    if isinstance(cell_type, str):
        if cell_type == 'lstm':
            return rnn.LSTMCell
        elif cell_type == 'gru':
            return rnn.GRUCell
        elif cell_type == 'relu_rnn':
            return partial(rnn.RNNCell, activation='relu')
        elif cell_type == 'tanh_rnn':
            return partial(rnn.RNNCell, activation='tanh')
        else:
            raise NotImplementedError
    else:
        return cell_type


def _get_attention_cell(attention_cell, units=None):
    """

    Parameters
    ----------
    attention_cell : AttentionCell or str
    units : int or None

    Returns
    -------
    attention_cell : AttentionCell
    """
    if isinstance(attention_cell, str):
        if attention_cell == 'scaled_luong':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           luong_style=True)
        elif attention_cell == 'scaled_dot':
            return DotProductAttentionCell(units=None, scaled=True, normalized=False,
                                           luong_style=False)
        elif attention_cell == 'dot':
            return DotProductAttentionCell(units=None, scaled=False, normalized=False,
                                           luong_style=False)
        elif attention_cell == 'cosine':
            return DotProductAttentionCell(units=units, scaled=False, normalized=True)
        elif attention_cell == 'mlp':
            return MLPAttentionCell(units=units, normalized=False)
        elif attention_cell == 'normed_mlp':
            return MLPAttentionCell(units=units, normalized=True)
        else:
            raise NotImplementedError
    else:
        assert isinstance(attention_cell, AttentionCell),\
            'attention_cell must be either string or AttentionCell. Received attention_cell={}'\
                .format(attention_cell)
        return attention_cell


def _nested_sequence_last(data, valid_length):
    """

    Parameters
    ----------
    data : nested container of NDArrays/Symbols
        The input data. Each element will have shape (batch_size, ...)
    valid_length : NDArray or Symbol
        Valid length of the sequences. Shape (batch_size,)
    Returns
    -------
    data_last: nested container of NDArrays/Symbols
        The last valid element in the sequence.
    """
    assert isinstance(data, list)
    if isinstance(data[0], (mx.sym.Symbol, mx.nd.NDArray)):
        F = mx.sym if isinstance(data[0], mx.sym.Symbol) else mx.ndarray
        return F.SequenceLast(F.stack(*data, axis=0),
                              sequence_length=valid_length,
                              use_sequence_length=True)
    elif isinstance(data[0], list):
        ret = []
        for i in range(len(data[0])):
            ret.append(_nested_sequence_last([ele[i] for ele in data], valid_length))
        return ret
    else:
        raise NotImplementedError


class Seq2SeqEncoder(Block):
    r"""Base class of the encoders in sequence to sequence learning models.
    """
    def __call__(self, inputs, valid_length=None, states=None):  #pylint: disable=arguments-differ
        """Encode the input sequence.

        Parameters
        ----------
        inputs : NDArray
            The input sequence, Shape (batch_size, length, C_in).
        valid_length : NDArray or None, default None
            The valid length of the input sequence, Shape (batch_size,). This is used when the
            input sequences are padded. If set to None, all elements in the sequence are used.
        states : list of NDArrays or None, default None
            List that contains the initial states of the encoder.

        Returns
        -------
        outputs : list
            Outputs of the encoder.
        """
        return super(Seq2SeqEncoder, self).__call__(inputs, valid_length, states)

    def forward(self, inputs, valid_length=None, states=None):  #pylint: disable=arguments-differ
        raise NotImplementedError


class Seq2SeqDecoder(Block):
    r"""Base class of the decoders in sequence to sequence learning models.

    In the forward function, it generates the one-step-ahead decoding output.

    """
    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length=None):
        r"""Generates the initial decoder states based on the encoder outputs.

        Parameters
        ----------
        encoder_outputs : list of NDArrays
        encoder_valid_length : NDArray or None

        Returns
        -------
        decoder_states : list
        """
        raise NotImplementedError

    def decode_seq(self, inputs, states, valid_length=None):
        r"""Given the inputs and the context computed by the encoder,
        generate the new states. This is usually used in the training phase where we set the inputs
        to be the target sequence.

        Parameters
        ----------
        inputs : NDArray
            The input embeddings. Shape (batch_size, length, C_in)
        states : list
            The initial states of the decoder.
        valid_length : NDArray or None
            valid length of the inputs. Shape (batch_size,)
        Returns
        -------
        output : NDArray
            The output of the decoder. Shape is (batch_size, length, C_out)
        states: list
            The new states of the decoder
        additional_outputs : list
            Additional outputs of the decoder, e.g, the attention weights
        """
        raise NotImplementedError

    def __call__(self, step_input, states):  #pylint: disable=arguments-differ
        r"""One-step decoding of the input

        Parameters
        ----------
        step_input : NDArray
            Shape (batch_size, C_in)
        states : list
            The previous states of the decoder
        Returns
        -------
        step_output : NDArray
            Shape (batch_size, C_out)
        states : list
        step_additional_outputs : list
            Additional outputs of the step, e.g, the attention weights
        """
        return super(Seq2SeqDecoder, self).__call__(step_input, states)

    def forward(self, step_input, states):  #pylint: disable=arguments-differ
        raise NotImplementedError


class NMTEncoder(Seq2SeqEncoder):
    r"""

    The encoder first stacks several bidirectional RNN layers and then stacks multiple
    uni-directional RNN layers with residual connections.

    Parameters
    ----------
    cell_type : str or function
        Can be "lstm", "gru" or constructor functions that can be directly called,
         like rnn.LSTMCell
    num_layers : int
        Total number of layers
    hidden_size : int
        Number of hidden units
    dropout : float
        The dropout rate
    bidirectional : bool
        Whether to use bidirectional or unidirectional RNN
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, cell_type='lstm', num_layers=3, hidden_size=300,
                 dropout=0.0, bidirectional=False,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 prefix=None, params=None):
        super(NMTEncoder, self).__init__(prefix=prefix, params=params)
        self._cell_type = _get_cell_type(cell_type)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._dropout = dropout
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.rnn_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.rnn_cells.add(
                    self._cell_type(hidden_size=self._hidden_size,
                                    i2h_weight_initializer=i2h_weight_initializer,
                                    h2h_weight_initializer=h2h_weight_initializer,
                                    i2h_bias_initializer=i2h_bias_initializer,
                                    h2h_bias_initializer=h2h_bias_initializer,
                                    prefix='rnn%d_' % i))

    def __call__(self, inputs, states=None, valid_length=None):
        """Encoder the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of initial states
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the last RNN layer
            - new_states of all the RNN layers
        """
        return super(NMTEncoder, self).__call__(inputs, states, valid_length)

    def forward(self, inputs, states=None, valid_length=None):  #pylint: disable=arguments-differ
        # TODO(sxjscience) Accelerate the forward using HybridBlock
        _, length, _ = inputs.shape
        new_states = []
        outputs = inputs
        for i, cell in enumerate(self.rnn_cells):
            begin_state = None if states is None else states[i]
            outputs, layer_states = cell.unroll(
                length=length, inputs=inputs, begin_state=begin_state, merge_outputs=True,
                valid_length=valid_length, layout='NTC')
            
            new_states.append(layer_states)
            # Apply Dropout
            outputs = self.dropout_layer(outputs)
            inputs = outputs

        if valid_length is not None:
            outputs = mx.nd.SequenceMask(outputs, sequence_length=valid_length,
                                         use_sequence_length=True, axis=1)
        return [outputs, new_states]


class NMTDecoder(HybridBlock, Seq2SeqDecoder):
    """Structure of the RNN Encoder 

    Parameters
    ----------
    cell_type : str or type
    attention_cell : AttentionCell or str
        Arguments of the attention cell.
        Can be 'scaled_luong', 'normed_mlp', 'dot'
    num_layers : int
    hidden_size : int
    dropout : float
    output_attention: bool
        Whether to output the attention weights
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, cell_type='lstm', attention_cell='scaled_luong',
                 num_layers=3, hidden_size=300,
                 dropout=0.0, output_attention=False,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 prefix=None, params=None):
        super(NMTDecoder, self).__init__(prefix=prefix, params=params)
        self._cell_type = _get_cell_type(cell_type)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._output_attention = output_attention
        with self.name_scope():
            self.attention_cell = _get_attention_cell(attention_cell, units=hidden_size)
            self.dropout_layer = nn.Dropout(dropout)
            self.rnn_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.rnn_cells.add(
                    self._cell_type(hidden_size=self._hidden_size,
                                    i2h_weight_initializer=i2h_weight_initializer,
                                    h2h_weight_initializer=h2h_weight_initializer,
                                    i2h_bias_initializer=i2h_bias_initializer,
                                    h2h_bias_initializer=h2h_bias_initializer,
                                    prefix='rnn%d_' % i))

    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length=None):
        """Initialize the state from the encoder outputs.

        Parameters
        ----------
        encoder_outputs : list -> [outputs, new_states]
        encoder_valid_length : NDArray or None

        Returns
        -------
        init_decoder_states : list
            The decoder states, includes:

            - rnn_states : NDArray
            - attention_vec : NDArray
            - mem_value : NDArray
            - mem_masks : NDArray, optional
        """
        mem_value, init_rnn_states = encoder_outputs
        batch_size, mem_length, mem_size = mem_value.shape
        init_attention_vec = mx.nd.zeros(shape=(batch_size, mem_size), ctx=mem_value.context)
        init_decoder_states = [init_rnn_states, init_attention_vec, mem_value]

        if encoder_valid_length is not None:
            '''
            mem_masks: (batch_size, mem_length)
            [
            [1, 1, 1, 1, 0, 0, ...] // with valid_length 1 s 
            [1, 1, 0, 0, 0, 0, ...]
            ...
            ]
            '''
            mem_masks = mx.nd.broadcast_lesser(
                mx.nd.arange(mem_length, ctx=encoder_valid_length.context).reshape((1, -1)),
                encoder_valid_length.reshape((-1, 1)))
            init_decoder_states.append(mem_masks)
        return init_decoder_states

    def decode_seq(self, inputs, states, valid_length=None):
        length = inputs.shape[1]
        rnn_output = []
        additional_outputs = []
        inputs = _as_list(mx.nd.split(inputs, num_outputs=length, axis=1, squeeze_axis=True))
        rnn_states = []
        attention_output = []
        fixed_states = states[2:] # [rnn_state, att_vec, mem_value, mem_masks]
        for i in range(length):
            ele_output, states, ele_additional_outputs = self.forward(inputs[i], states)
            rnn_states.append(states[0])
            attention_output.append(states[1])
            rnn_output.append(ele_output)
            additional_outputs.append(ele_additional_outputs)

        if valid_length is not None:
            states = [_nested_sequence_last(rnn_states, valid_length),
                      _nested_sequence_last(attention_output, valid_length)] + fixed_states

        attention_output = mx.nd.stack(*attention_output, axis=1)
        rnn_output = mx.nd.stack(*rnn_output, axis=1)
        combined_output = mx.nd.concat(rnn_output, attention_output, dim=2)
        if valid_length is not None:
            combined_output = mx.nd.SequenceMask(combined_output,
                                        sequence_length=valid_length,
                                        use_sequence_length=True,
                                        axis=1)
        if self._output_attention:
            additional_outputs = [mx.nd.concat(*additional_outputs, dim=-2)]
        return combined_output, states, additional_outputs

    def __call__(self, step_input, states):
        """One-step-ahead decoding of the GNMT decoder.

        Parameters
        ----------
        step_input : NDArray or Symbol
        states : NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size, C_out)
        new_states: list
            Includes

            - rnn_states : list of NDArray or Symbol
            - attention_vec : NDArray or Symbol, Shape (batch_size, C_memory)
            - mem_value : NDArray
            - mem_masks : NDArray, optional

        step_additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, 1, mem_length) or
            (batch_size, num_heads, 1, mem_length)
        """
        return super(NMTDecoder, self).__call__(step_input, states)

    def forward(self, step_input, states):  #pylint: disable=arguments-differ
        step_output, new_states, step_additional_outputs =\
            super(NMTDecoder, self).forward(step_input, states)
        # In hybrid_forward, only the rnn_states and attention_vec are calculated.
        # We directly append the mem_value and mem_masks in the forward() function.
        # We apply this trick because the memory value/mask can be directly appended to the next
        # timestamp and there is no need to create additional NDArrays. If we use HybridBlock,
        # new NDArrays will be created even for identity mapping.
        # See https://github.com/apache/incubator-mxnet/issues/10167
        new_states += states[2:]
        return step_output, new_states, step_additional_outputs

    def hybrid_forward(self, F, step_input, states):  #pylint: disable=arguments-differ
        """

        Parameters
        ----------
        step_input : NDArray or Symbol
        states : NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size, C_out)
        new_states: list
            Includes

            - rnn_states : list of NDArray or Symbol
            - attention_vec : NDArray or Symbol, Shape (batch_size, C_memory)

        step_additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, 1, mem_length) or
            (batch_size, num_heads, 1, mem_length)

        """
        has_mem_mask = (len(states) == 4)
        if has_mem_mask:
            rnn_states, attention_output, mem_value, mem_masks = states
            mem_masks = F.expand_dims(mem_masks, axis=1)
        else:
            rnn_states, attention_output, mem_value = states
            mem_masks = None
        
        new_rnn_states = []
        curr_input = step_input
        # Process the 1nd layer - the last layer
        for i in range(0, len(self.rnn_cells)):
            rnn_out, layer_state = self.rnn_cells[i](curr_input, rnn_states[i])
            rnn_out = self.dropout_layer(rnn_out)
            # if self._use_residual:
            #     rnn_out = rnn_out + curr_input
            # Append new RNN state
            curr_input = rnn_out
            new_rnn_states.append(layer_state)

        # calculate the attention_vec
        attention_vec, attention_weights =\
            self.attention_cell(F.expand_dims(rnn_out, axis=1),  # Shape(B, 1, C)
                                mem_value,
                                mem_value,
                                mem_masks)
        attention_vec = F.reshape(attention_vec, shape=(0, -1))

        new_states = [new_rnn_states, attention_vec]
        step_additional_outputs = None
        if self._output_attention:
            step_additional_outputs = attention_weights
        return rnn_out, new_states, step_additional_outputs


def get_nmt_encoder_decoder(cell_type='lstm', attention_cell='scaled_luong', num_layers=3,
                             hidden_size=300, dropout=0.0, bidirectional=False,
                             i2h_weight_initializer=None, h2h_weight_initializer=None,
                             i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                             prefix='gnmt_', params=None):
    """Build a pair of GNMT encoder/decoder

    Parameters
    ----------
    cell_type : str or type
    attention_cell : str or AttentionCell
    num_layers : int
    hidden_size : int
    dropout : float
    bidirectional : bool
    i2h_weight_initializer : mx.init.Initializer or None
    h2h_weight_initializer : mx.init.Initializer or None
    i2h_bias_initializer : mx.init.Initializer or None
    h2h_bias_initializer : mx.init.Initializer or None
    prefix :
    params :

    Returns
    -------
    encoder : NMTEncoder
    decoder : NMTDecoder
    """
    encoder = NMTEncoder(cell_type=cell_type, num_layers=num_layers,
                          hidden_size=hidden_size, dropout=dropout,
                          bidirectional=bidirectional,
                          i2h_weight_initializer=i2h_weight_initializer,
                          h2h_weight_initializer=h2h_weight_initializer,
                          i2h_bias_initializer=i2h_bias_initializer,
                          h2h_bias_initializer=h2h_bias_initializer,
                          prefix=prefix + 'enc_', params=params)
    decoder = NMTDecoder(cell_type=cell_type, attention_cell=attention_cell, num_layers=num_layers,
                          hidden_size=hidden_size, dropout=dropout,
                          i2h_weight_initializer=i2h_weight_initializer,
                          h2h_weight_initializer=h2h_weight_initializer,
                          i2h_bias_initializer=i2h_bias_initializer,
                          h2h_bias_initializer=h2h_bias_initializer,
                          prefix=prefix + 'dec_', params=params)
    return encoder, decoder
