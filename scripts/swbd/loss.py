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
"""Loss functions."""

import numpy as np
from mxnet.gluon.loss import SoftmaxCELoss
from mxnet.gluon.loss import Loss, _apply_weighting


class SoftmaxCEMaskedLoss(SoftmaxCELoss):
    """Wrapper of the SoftmaxCELoss that supports valid_length as the input

    """
    def hybrid_forward(self, F, pred, label, valid_length):
        """

        Parameters
        ----------
        F
        pred : Symbol or NDArray
            Shape (batch_size, length, V)
        label : Symbol or NDArray
            Shape (batch_size, length)
        valid_length : Symbol or NDArray
            Shape (batch_size, )
        Returns
        -------
        loss : Symbol or NDArray
            Shape (batch_size,)
        """
        sample_weight = F.cast(F.expand_dims(F.ones_like(label), axis=-1), dtype=np.float32)
        sample_weight = F.SequenceMask(sample_weight,
                                       sequence_length=valid_length,
                                       use_sequence_length=True,
                                       axis=1)
        return super(SoftmaxCEMaskedLoss, self).hybrid_forward(F, pred, label, sample_weight)

class CtcLoss(Loss):
    r"""Connectionist Temporal Classification Loss.


    Parameters
    ----------
    layout : str, default 'NTC'
        Layout of prediction tensor. 'N', 'T', 'C' stands for batch size,
        sequence length, and alphabet_size respectively.
    label_layout : str, default 'NT'
        Layout of the labels. 'N', 'T' stands for batch size, and sequence
        length respectively.
    weight : float or None
        Global scalar weight for loss.


    Inputs:
        - **pred**: unnormalized prediction tensor (before softmax).
          Its shape depends on `layout`. If `layout` is 'TNC', pred
          should have shape `(sequence_length, batch_size, alphabet_size)`.
          Note that in the last dimension, index `alphabet_size-1` is reserved
          for internal use as blank label. So `alphabet_size` is one plus the
          actual alphabet size.

        - **label**: zero-based label tensor. Its shape depends on `label_layout`.
          If `label_layout` is 'TN', `label` should have shape
          `(label_sequence_length, batch_size)`.

        - **pred_lengths**: optional (default None), used for specifying the
          length of each entry when different `pred` entries in the same batch
          have different lengths. `pred_lengths` should have shape `(batch_size,)`.

        - **label_lengths**: optional (default None), used for specifying the
          length of each entry when different `label` entries in the same batch
          have different lengths. `label_lengths` should have shape `(batch_size,)`.

    Outputs:
        - **loss**: output loss has shape `(batch_size,)`.


    **Example**: suppose the vocabulary is `[a, b, c]`, and in one batch we
    have three sequences 'ba', 'cbb', and 'abac'. We can index the labels as
    `{'a': 0, 'b': 1, 'c': 2, blank: 3}`. Then `alphabet_size` should be 4,
    where label 3 is reserved for internal use by `CTCLoss`. We then need to
    pad each sequence with `-1` to make a rectangular `label` tensor::

        [[1, 0, -1, -1],
         [2, 1,  1, -1],
         [0, 1,  0,  2]]


    References
    ----------
        `Connectionist Temporal Classification: Labelling Unsegmented
        Sequence Data with Recurrent Neural Networks
        `_
    """
    def __init__(self, layout='NTC', label_layout='NT', weight=None, **kwargs):
        assert layout in ['NTC', 'TNC'],\
               "Only 'NTC' and 'TNC' layouts for pred are supported. Got: %s"%layout
        assert label_layout in ['NT', 'TN'],\
               "Only 'NT' and 'TN' layouts for label are supported. Got: %s"%label_layout
        self._layout = layout
        self._label_layout = label_layout
        batch_axis = label_layout.find('N')
        super(CTCLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label,
                       pred_lengths=None, label_lengths=None, sample_weight=None, blank_label='first'):
        if self._layout == 'NTC':
            pred = F.swapaxes(pred, 0, 1)
        if self._batch_axis == 1:
            label = F.swapaxes(label, 0, 1)
        loss = F.contrib.CTCLoss(pred, label, pred_lengths, label_lengths,
                                 use_data_lengths=pred_lengths is not None,
                                 use_label_lengths=label_lengths is not None,
                                 blank_label=blank_label)
        return _apply_weighting(F, loss, self._weight, sample_weight)