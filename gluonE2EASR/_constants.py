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

# pylint: disable=too-many-lines
"""Constants."""

UNK_TOKEN = '<unk>'

BOS_TOKEN = '<bos>'

EOS_TOKEN = '<eos>'

PAD_TOKEN = '<pad>'

UNK_IDX = 0   # This should not be changed as long as serialized token
              # embeddings redistributed on S3 contain an unknown token.
              # Blame this code change and see commit for more context.

LARGE_POSITIVE_FLOAT = 1e18

LARGE_NEGATIVE_FLOAT = -LARGE_POSITIVE_FLOAT
