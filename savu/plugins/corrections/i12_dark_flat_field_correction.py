# Copyright 2014 Diamond Light Source Ltd.
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

"""
.. module:: i12_dark_flat_field_correction
   :platform: Unix
   :synopsis: A Plugin to apply a simple dark and flatfield correction to some
       raw timeseries data

.. moduleauthor:: Nicola Wadeson <scientificsoftware@diamond.ac.uk>

"""

from savu.plugins.driver.cpu_plugin import CpuPlugin
from savu.plugins.base_correction import BaseCorrection

import numpy as np

from savu.plugins.utils import register_plugin


@register_plugin
class I12DarkFlatFieldCorrection(BaseCorrection, CpuPlugin):
    """
    A Plugin to apply a simple dark and flatfield correction to i12 projection
    data.
    """

    def __init__(self):
        super(I12DarkFlatFieldCorrection,
              self).__init__("I12DarkFlatFieldCorrection")

    def pre_process(self):
        self.dark = self.exp.meta_data.get_meta_data('dark')
        self.flat = self.exp.meta_data.get_meta_data('flat')
        #reduce data to the correct size here

    def correct(self, data):
        print data.shape, self.dark.shape, self.flat.shape
        return data

    def setup(self):
        """
        Initial setup of all datasets required as input and output to the
        plugin.  This method is called before the process method in the plugin
        chain.
        """
        # set up the output dataset that is created by the plugin
        in_dataset, out_dataset = self.get_datasets()
        # copy all required information from in_dataset[0]
        out_dataset[0].create_dataset(in_dataset[0])

        # set information relating to the plugin data
        in_pData, out_pData = self.get_plugin_datasets()
        # set pattern_name and nframes to process for all datasets
        in_pData[0].plugin_data_setup('PROJECTION', self.get_max_frames())
        out_pData[0].plugin_data_setup('PROJECTION', self.get_max_frames())
        print "^^^^^^^^^^^^^^^^^^^^^", in_dataset[0].get_shape(), out_dataset[0].get_shape()

    def get_max_frames(self):
        return 4
