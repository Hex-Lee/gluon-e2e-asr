'''encoder and  decoder setup '''
from functools import partial
import mxnet as mx
from mxnet.base import _as_list
from mxnet.gluon import nn, rnn
from mxnet.gluon.block import Block, HybridBlock
from gluonE2EASR.model.base_encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder
from gluonE2EASR.model.attention_cell import AttentionCell, MLPAttentionCell, DotProductAttentionCell

def _get_cell_type(cell_type):
    if isinstance(cell_type, str):
        if cell_type == 'lstm':
            return rnn.LSTM
        elif cell_type == 'gru':
            return rnn.GRU
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


    '''scaled: bool, default True
                    Whether to divide the attention weights by the sqrt of the query dimension.
                    This is first proposed in "[NIPS2017] Attention is all you need."::

                        score = <h_q, h_k> / sqrt(dim_q)'''

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

def _nested_copy_states(states, num_layers, bidirectional=False):
    '''
    Parameters
    ----------
    states : list of NDArrays/Symbols or NDArrays/Symbols
        The last valid encoder rnn state in the sequence. 
        For lstm, it should be a list of two initial recurrent state tensors [c, h]
        Each has shape (num_layers, batch_size, num_hidden)

        Notes: since we forward the padded sequence at one time, the last_sate
               will contain the padding forward results 
    num_layers : int
        The output number layers of the state

    Returns 
    -------
    out_states :  list of NDArrays/Symbols or NDArrays/Symbols
        The same shape as the states, orther than has the num_layers 
    '''
    if isinstance(states, (mx.sym.Symbol, mx.nd.NDArray)):
        F = mx.sym if isinstance(states, mx.sym.Symbol) else mx.ndarray
        input_layers = states.shape[0]
        #print(input_layers, num_layers, states.shape)
        if bidirectional:
            last_layer_states = F.concat(F.expand_dims(states[-1], axis=0),
                                        F.expand_dims(states[-2], axis=0),
                                        dim=2)
        else:
            last_layer_states = F.expand_dims(states[-1], axis=0) # (1, B, H)
        
        #Just copy the last layer(for blstm, concat the l->, r-> direction)
        return F.broadcast_axis(last_layer_states, axis=0, size=num_layers)
        #Copy the same layer states !!!
        # if input_layers > num_layers:
        #     return states[input_layers-num_layers:]
        # elif input_layers < num_layers:
        #     return F.concat(states, 
        #                     F.broadcast_axis(last_layer_states, axis=0, 
        #                                     size=num_layers-input_layers)
        #                     )
        # else:
        #     return states
    elif isinstance(states, list):
        ret = []
        for ele in states:
            ret.append(_nested_copy_states(ele, num_layers, bidirectional))
        return ret
    else:
        raise NotImplementedError

class SimpleEncoder(HybridBlock):
    def __init__(self, cell_type='lstm', num_layers=3, hidden_size=300, 
                 dropout=0.0, bidirectional=False, prefix=None, params=None):
        super(SimpleEncoder, self).__init__(prefix=prefix, params=params)
        self._cell_type = _get_cell_type(cell_type)
        self._num_layers = num_layers
        self._bidirectional = bidirectional
        if bidirectional:
            hidden_size = hidden_size // 2
        self._hidden_size = hidden_size
        with self.name_scope():
            self.rnn_cells = self._cell_type(hidden_size=hidden_size,
                                             num_layers=num_layers,
                                             layout='NTC',
                                             dropout=dropout,
                                             bidirectional=bidirectional)

    def hybrid_forward(self, F, inputs, valid_length):
        """

        Parameters
        ----------
        inputs :  NDArrays or Symbols
            The input data. Shape (batch_size, sequence_length, in_units)
        valid_length : NDArray or Symbol
            Valid length of the sequences. Shape (batch_size,)
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
        batch_size = inputs.shape[0]
        begin_states = self.rnn_cells.begin_state(batch_size, ctx=inputs.context)
        # thy states will always be a list, [ele, ], each ele has 
        # the Shape (num_layers, batch_size, hidden_sizes)
        outputs, last_states = self.rnn_cells(inputs, begin_states)
        outputs = F.SequenceMask(outputs, sequence_length=valid_length, 
                                        use_sequence_length=True, axis=1)
        return outputs, last_states

class SimpleDecoder(HybridBlock):
    def __init__(self, cell_type='lstm', attention_cell='scaled_luong',
                 num_layers=3, hidden_size=300, dropout=0.0, 
                 bidirectional=False, prefix=None, params=None):
        super(SimpleDecoder, self).__init__(prefix=prefix, params=params)
        self._cell_type = _get_cell_type(cell_type)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        with self.name_scope():
            self.attention_cell = _get_attention_cell(attention_cell, units=hidden_size)
            self.rnn_cells = self._cell_type(hidden_size=hidden_size,
                                             num_layers=num_layers,
                                             layout='NTC',
                                             dropout=dropout,
                                             bidirectional=bidirectional)

    def init_state_from_encoder(self, mem_value, enc_rnn_states, encoder_valid_length=None, enc_bidirectional=False):
        """Initialize the state from the encoder outputs and states.

        Parameters
        ----------
        mem_value : NDArray or None (batch_size, enc_seq_length, mem_size)
            The encoding memory of the sequence, Generally are outputs of the encoder
        enc_rnn_states : list of NDArrays/Symbols or NDArrays/Symbols
            The last valid encoder rnn state in the sequence. 
            For lstm, it should be a list of two initial recurrent state tensors [c, h]
            Each has shape (num_layers, batch_size, num_hidden)
            
            Notes: since we forward the padded sequence at one time, the last_sate
                   will contain the padding forward results     
        encoder_valid_length : NDArray or None

        Returns
        -------
        init_decoder_states : list
            The decoder states, includes:
            - rnn_states : list of NDArrays/Symbols or NDArrays/Symbols
            - mem_value : NDArray
            - mem_masks : NDArray, optional
        """        
        batch_size, mem_length, mem_size = mem_value.shape

        # use the last layer hidden states of the encoder as the decoder begin states
        init_dec_rnn_states = _nested_copy_states(enc_rnn_states, self._num_layers, enc_bidirectional)
        init_decoder_states = [init_dec_rnn_states, mem_value]
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

    def decode_seq(self, inputs, decoder_states, valid_length=None):
        """decoding of the whole sequence.

        Parameters
        ----------
        inputs : NDArray or Symbol 
            decoder sequence input. Shape: (batch_size, sequence_length, in_units) 
        decoder_states: list
            Includes
            - rnn_states : list of NDArray or Symbol 
            - mem_value : NDArray
            - mem_masks : NDArray, optional

        valid_length: NDArray or Symbol 
            target valid length. Shape (batch_size, )

        Returns
        -------
        rnn_outputs : NDArray or Symbol
            The output of the decoder. Shape is (batch_size, sequence_length, units)
        context_vecs: NDArray or Symbol
            The attention contexts. Shape: (batch_size, sequence_length, mem_size)
        attention_weights: NDArray or Symbol
            The attention weights. Shape: (batch_size, sequence_length, mem_sequence_length)
        """
        if len(decoder_states) == 3:
            rnn_states, mem_value, mem_masks = decoder_states
            mem_masks = mx.nd.expand_dims(mem_masks, axis=1)
            mem_masks = mx.nd.broadcast_axis(mem_masks, axis=1, size=inputs.shape[1])
        else:
            rnn_states, mem_value = decoder_states
            mem_masks = None
        # the hidden still take the padding value into counts      
        rnn_outputs, rnn_states = self.rnn_cells(inputs, rnn_states)
        if self._bidirectional:
            rnn_outputs = rnn_outputs[:,:,:self._hidden_size]

        context_vecs, attention_weights = self.attention_cell(rnn_outputs,  # Shape(N, L, C)
                                                              mem_value,    # Shaop(N, T, C)
                                                              mem_value,
                                                              mem_masks)    # Shape(N, L, T)

        valid_length = None ## we can just use valid length in Loss function
        if valid_length is not None:
            rnn_outputs = mx.nd.SequenceMask(rnn_outputs, sequence_length=valid_length, 
                                            use_sequence_length=True, axis=1)
            context_vecs = mx.nd.SequenceMask(context_vecs, sequence_length=valid_length, 
                                            use_sequence_length=True, axis=1)
            attention_weights = mx.nd.SequenceMask(attention_weights, sequence_length=valid_length, 
                                            use_sequence_length=True, axis=1)

        return rnn_outputs, context_vecs, attention_weights

    def __call__(self, step_input, decoder_states):
        """One-step-ahead decoding of the GNMT decoder.

        Parameters
        ----------
        step_input : NDArray or Symbol 
            one step of decoder input. Shape: (batch_size, in_units) 
        decoder_states: list
            Includes
                - rnn_states : list of NDArray or Symbol
                - mem_value : NDArray
                - mem_masks : NDArray, optional

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size, C_out)
        new_decoder_states: list
            Includes
                - rnn_states : list of NDArray or Symbol
                - mem_value : NDArray
                - mem_masks : NDArray, optional
        attention_weight: NDArray or Symbol
            The attention weight. Shape: (batch_size, mem_sequence_length)
        """
        return self.forward(step_input, decoder_states)

    def forward(self, step_input, decoder_states):  #pylint: disable=arguments-differ
        step_output, new_decoder_states, context_vec, attention_weight =\
            super(SimpleDecoder, self).forward(step_input, decoder_states) 
        # the father class will be HybridBlock, which will call the hybrid_forward
        # In hybrid_forward, only the rnn_states and context_vec are calculated.
        # We directly append the mem_value and mem_masks in the forward() function.
        # We apply this trick because the memory value/mask can be directly appended to the next
        # timestamp and there is no need to create additional NDArrays. If we use HybridBlock,
        # new NDArrays will be created even for identity mapping.
        # See https://github.com/apache/incubator-mxnet/issues/10167
        new_decoder_states += decoder_states[1:]

        return step_output.reshape(shape=(0,-1)), new_decoder_states,\
                context_vec.reshape(shape=(0,-1)), attention_weight.reshape(shape=(0,-1))

    def hybrid_forward(self, F, step_input, decoder_states):  #pylint: disable=arguments-differ
        if len(decoder_states) == 3:
            rnn_states, mem_value, mem_masks = decoder_states
            mem_masks = F.expand_dims(mem_masks, axis=1)
        else:
            rnn_states, mem_value = decoder_states
            mem_masks = None

        # Process the rnn cells
        rnn_out, rnn_states = self.rnn_cells(F.expand_dims(step_input, axis=1),  # Shape(B, 1, C)
                                             rnn_states)
        # calculate the context_vec
        context_vec, attention_weight =\
                        self.attention_cell(rnn_out,  # Shape(B, 1, C)
                                            mem_value,
                                            mem_value,
                                            mem_masks)
            
        context_vec = F.reshape(context_vec, shape=(0, -1))
        new_decode_states = [rnn_states]

        return rnn_out, new_decode_states, context_vec, attention_weight


def get_simple_encoder_decoder(cell_type='lstm', attention_cell='dot', 
                                num_enc_layers=3, enc_bidirectional=False, 
                                num_dec_layers=3, dec_bidirectional=False, 
                                hidden_size=300, dropout=0.0, 
                                prefix='gnmt_', params=None):
    """Build a pair of GNMT encoder/decoder

    Parameters
    ----------
    cell_type : str or type
    attention_cell : str or AttentionCell
    num_enc_layers: int
    num_enc_bi_layers: int
    num_dec_layers : int
    hidden_size : int
    dropout : float
    use_residual : bool
    prefix :
    params :

    Returns
    -------
    encoder : SimpleEncoder
    decoder : SimpleDecoder
    """
    encoder = SimpleEncoder(cell_type=cell_type, num_layers=num_enc_layers, hidden_size=hidden_size, 
                             dropout=dropout, bidirectional=enc_bidirectional, 
                            prefix='enc_', params=None)
    decoder = SimpleDecoder(cell_type=cell_type, attention_cell=attention_cell,
                             num_layers=num_dec_layers, hidden_size=hidden_size, 
                             dropout=dropout, bidirectional=dec_bidirectional, 
                             prefix='dec_', params=None)
    return encoder, decoder
