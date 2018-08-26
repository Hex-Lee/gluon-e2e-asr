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

# pylint: disable=consider-iterating-dictionary

"""Vocabulary."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['Vocab']

import json
import warnings
import logging

from mxnet import nd

from .data.utils import DefaultLookupDict
from . import _constants as C

logger = logging.getLogger(__name__)

class Vocab(object):
    def __init__(self, vocab_file=None, 
                 unknown_token=C.UNK_TOKEN, padding_token=C.PAD_TOKEN, 
                 bos_token=C.BOS_TOKEN, eos_token=C.EOS_TOKEN,
                 reserved_tokens=None):

        self._unknown_token = unknown_token
        self._padding_token = padding_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        
        special_tokens = []
        if padding_token:
            special_tokens.append(padding_token)
        if bos_token:
            special_tokens.append(bos_token)
        if eos_token:
            special_tokens.append(eos_token)
        if reserved_tokens:
            special_tokens.extend(reserved_tokens)
            special_token_set = set(special_tokens)
            if unknown_token:
                assert unknown_token not in special_token_set, \
                    '`reserved_token` cannot contain `unknown_token`.'
            assert len(special_token_set) == len(special_tokens), \
                '`reserved_tokens` cannot contain duplicate reserved tokens or ' \
                'other special tokens.'

        if not special_tokens:
            self._reserved_tokens = None
        else:
            self._reserved_tokens = special_tokens[:]

        self._idx_to_token = []
        self._token_to_idx = DefaultLookupDict()

        asset vocab_file is not None, 'lack of vocabulary file!'
        logger.info("Use the custom vocab_file: {}".format(vocab_file))
        self._load_vocab_file_with_index(vocab_file)      
        self._index_special_tokens(unknown_token, special_tokens)

    def _load_vocab_file_with_index(self, vocab_file):
        """load the vocabulary files.

            vocab_file will have the format(index start from 0):
                word1 0
                word2 1
                ...
        """
        with open(vocab_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip().split()
                token = line[0]
                index = int(line[1])
                assert index==i, 'the vocab file index is not continuous, lack of {}'.format(i)
                self._idx_to_token.append(token)
                self._token_to_idx[token] = index

    def _index_special_tokens(self, unknown_token, special_tokens):
        """Indexes unknown and reserved tokens."""

        if unknown_token:
            unknown_and_special_tokens = special_tokens + [unknown_token]
        else:
            unknown_and_special_tokens = special_tokens
            
        for t in unknown_and_special_tokens:
            if not t in self._token_to_idx:
                self._idx_to_token.append(t)
                self._token_to_idx[t] = len(self._idx_to_token) - 1

        if unknown_token:
            unk_index = self._token_to_idx[unknown_token]
            self._token_to_idx._set_default_lookup_value(unk_index)

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def reserved_tokens(self):
        return self._reserved_tokens

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def padding_token(self):
        return self._padding_token

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    def __contains__(self, token):
        """Checks whether a text token exists in the vocabulary.


        Parameters
        ----------
        token : str
            A text token.


        Returns
        -------
        bool
            Whether the text token exists in the vocabulary (including `unknown_token`).
        """

        return token in self._token_to_idx

    def __getitem__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.

        If `unknown_token` of the vocabulary is None, looking up unknown tokens results in KeyError.

        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[tokens]
        else:
            return [self._token_to_idx[token] for token in tokens]

    def __len__(self):
        return len(self._idx_to_token)

    def to_tokens(self, indices):
        """Converts token indices to tokens according to the vocabulary.


        Parameters
        ----------
        indices : int or list of ints
            A source token index or token indices to be converted.


        Returns
        -------
        str or list of strs
            A token or a list of tokens according to the vocabulary.
        """

        to_reduce = False
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
            to_reduce = True

        max_idx = len(self._idx_to_token) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx:
                raise ValueError('Token index {} in the provided `indices` is invalid.'.format(idx))
            else:
                tokens.append(self._idx_to_token[idx])

        return tokens[0] if to_reduce else tokens

    def to_indices(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        return self[tokens]

    def __call__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        return self[tokens]

    def __repr__(self):
        return 'Vocab(size={}, unk="{}", reserved="{}")'.format(len(self), self._unknown_token,
                                                                self._reserved_tokens)

    def to_json(self):
        """Serialize Vocab object to json string.
        """
        vocab_dict = {}
        vocab_dict['idx_to_token'] = self._idx_to_token
        vocab_dict['token_to_idx'] = dict(self._token_to_idx)
        vocab_dict['reserved_tokens'] = self._reserved_tokens
        vocab_dict['unknown_token'] = self._unknown_token
        vocab_dict['padding_token'] = self._padding_token
        vocab_dict['bos_token'] = self._bos_token
        vocab_dict['eos_token'] = self._eos_token
        return json.dumps(vocab_dict)

    @staticmethod
    def from_json(json_str):
        """Deserialize Vocab object from json string.

        Parameters
        ----------
        json_str : str
            Serialized json string of a Vocab object.


        Returns
        -------
        Vocab
        """
        vocab_dict = json.loads(json_str)

        unknown_token = vocab_dict.get('unknown_token')
        vocab = Vo
        cab(unknown_token=unknown_token)
        vocab._idx_to_token = vocab_dict.get('idx_to_token')
        vocab._token_to_idx = vocab_dict.get('token_to_idx')
        if unknown_token:
            vocab._token_to_idx = DefaultLookupDict(vocab._token_to_idx[unknown_token],
                                                    vocab._token_to_idx)
        vocab._reserved_tokens = vocab_dict.get('reserved_tokens')
        vocab._padding_token = vocab_dict.get('padding_token')
        vocab._bos_token = vocab_dict.get('bos_token')
        vocab._eos_token = vocab_dict.get('eos_token')
        return vocab
