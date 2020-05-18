# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dummy data sets used for testing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.data.ground_truth import ground_truth_data
import json
import numpy as np
import os
from disentanglement_lib.data.ground_truth import utils

CELEBA_PATH = "/hdd_c/data/img_align_celeba"


class CelebA(ground_truth_data.GroundTruthData):
  """Dummy image data set of random noise used for testing."""

  def __init__(self):
    print('Hello!')
    self.factor_updated = False
    self.num_samples = 100000
    self.images = self._load_data()

  @property
  def num_factors(self):
    return 5

  @property
  def observation_shape(self):
    return [64, 64, 3]

  def _load_data(self):
    TRAIN_STOP = self.num_samples
    self._TRAIN_STOP = TRAIN_STOP
    print(TRAIN_STOP)
    print(CELEBA_PATH)
    celebA = [os.path.join(root, filename)
      for root, dirnames, filenames in os.walk(CELEBA_PATH)
      for filename in filenames if filename.endswith('.jpg')]
    celebA=celebA[:TRAIN_STOP]    
    images = [utils.get_image(name,
                    input_height=178,
                    input_width=218,
                    resize_height=64,
                    resize_width=64,
                    is_crop=True)/255. for name in celebA]
    print('finish reading face images')
    images = np.asarray(images)
    print(images.shape)
    return images

  def _load_batch_data(self, indices):
    filenames = [os.path.join(CELEBA_PATH, f'{ind+1:06}.jpg') for ind in indices]
    images = [utils.get_image(name,
                    input_height=178,
                    input_width=218,
                    resize_height=64,
                    resize_width=64,
                              is_crop=True)/255. for name in filenames]
    images = np.asarray(images)
#    print(images.shape)
    return images

  def sample_factors(self, num, random_state):
    self.indices = random_state.randint(self.num_samples, size=num)
    self.factor_updated = True
    return None

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    if not self.factor_updated:
        raise NotImplementedError
    self.factor_updated = False
    obs = self._load_batch_data(self.indices) #self.images[self.indices]
    return obs
