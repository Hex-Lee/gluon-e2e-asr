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

"""Utility classes and functions. They help organize and keep statistics of datasets."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['DefaultLookupDict']

class DefaultLookupDict(dict):
    """Dictionary class with fall-back look-up with default value set in the constructor."""

    def __init__(self, default=-1, d=None):
        if d:
            super(DefaultLookupDict, self).__init__(d)
        else:
            super(DefaultLookupDict, self).__init__()
        self._default = default

    def _set_default_lookup_value(self, default):
        self._default = default

    def __getitem__(self, k):
        return self.get(k, self._default)
