# Copyright 2019 Diamond Light Source Ltd.
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
.. module:: Segmentation pipeline for i23 reconstructed data based on morphological level sets and geodesic distances
   :platform: Unix
   :synopsis: Wrapper for i23 segmentation pipeline

.. moduleauthor:: Daniil Kazantsev <scientificsoftware@diamond.ac.uk>
"""

from savu.plugins.plugin import Plugin
from savu.plugins.driver.multi_threaded_plugin import MultiThreadedPlugin
from savu.plugins.utils import register_plugin

import numpy as np
# Here is a list of required software
# Geodesic distance transform, the software can be installed from https://github.com/taigw/geodesic_distance
import geodesic_distance
# Morphological snakes or level sets, can be installed from  https://github.com/pmneila/morphsnakes
from morphsnakes import morphological_chan_vese, circle_level_set
# CCPi Regularisation Toolkit, https://github.com/vais-ral/CCPi-Regularisation-Toolkit
from ccpi.filters.regularisers import SB_TV
# Morphological operations from skimage
from skimage.morphology import opening, closing
from skimage.morphology import disk


def initialiseLS(slices, NxSize, NySize, coordX0, coordY0, coordZ0, coordX1, coordY1, coordZ1, circle_size):
    LS_init = np.uint8(np.zeros((slices, NxSize, NySize)))
    # calculate coordinates
    steps = coordZ1 - coordZ0
    if ((steps <= 0) or (steps > slices)):
        raise Exception("Z coordinates are given incorrectly (out of array range)")
    distance = np.sqrt((coordX1 - coordX0)**2 + (coordY1 - coordY0)**2)
    d_dist = distance/(steps-1)
    d_step = d_dist
    
    for j in range(coordZ0,coordZ1):
        t = d_step/distance
        x_t = np.round((1.0 - t)*coordX0 + t*coordX1)
        y_t = np.round((1.0 - t)*coordY0 + t*coordY1)
        if (coordX0 == coordX1):
            x_t = coordX0
        if(coordY0 == coordY1):
            y_t = coordY0
        LS_init[j,:,:] = circle_level_set(tuple((NxSize, NySize)), (y_t, x_t), circle_size)
        d_step += d_dist
    return LS_init

def morphological_proc(data, disk_size):
    selem = disk(disk_size)
    Morph = np.uint8(np.zeros(np.shape(data)))
    slices, NxSize, NySize = np.shape(data)
    for j in range(0,slices):
        segm_to_proc = data[j,:,:].copy()
        closing_t = closing(segm_to_proc, selem)
        segm_tmp = opening(closing_t, selem)
        Morph[j,:,:] = segm_tmp
    return Morph

@register_plugin
class I23SegmentLs(Plugin, MultiThreadedPlugin):
    """
    A Plugin to segment data using a combination of consequential methods:
        1. User must initialise the process giving the coordinates of the crystal (approximate) central point in XY (vertical) slicing. 
        2. Based on the initialisation the geodesic distances (raster scan) will be calculated
        3. The result will be 3D denoised by Split Bregman TV
        4. Morphological level sets will be started using the result of (3) to get crystal segmented and mild morphological processing of it
        5. Morphological level sets initialised with the result of (4) to get liqeur segmented
        6. Morphological level sets initialised with the result of (5) to get the loop
        7. Morphological level sets initialised with the result of (6) to get the whole object

    :param coordinates : Crystal X-Y  coordinates in XY slicing  . Default: [300, 300, 310, 310].
    :param regularisationTV: Regularisation parameter for TV denoising. Default: 0.033.
    :param crystal_lambda: segmentation of crystal as CV parameter. Default: 0.0035.
    :param liquor_lambda: segmentation of liquor as CV parameter. Default: 0.045.
    :param whole_lambda: segmentation of the whole object as CV parameter. Default: 0.48.
    :param chanvese_iterations: The number of Chan-Vese iterations. Default: 350.
    :param disk_size: disk size for morphological post processing. Default: 7.
    """

    def __init__(self):
        super(I23SegmentLs, self).__init__("I23SegmentLs")

    def setup(self):
    
        in_dataset, out_dataset = self.get_datasets()
        out_dataset[0].create_dataset(in_dataset[0], dtype=np.uint8)
        in_pData, out_pData = self.get_plugin_datasets()
        in_pData[0].plugin_data_setup('VOLUME_3D', 'single')
        out_pData[0].plugin_data_setup('VOLUME_3D', 'single')
        
    def pre_process(self):
        # extract given parameters
        self.classes = self.parameters['classes']

    def process_frames(self, data):
        # Do GMM classification/segmentation first
        dimensdata = data[0].ndim
        if (dimensdata == 2):
            (Nsize1, Nsize2) = np.shape(data[0])
            Nsize3 = 1
        if (dimensdata == 3):
            (Nsize1, Nsize2, Nsize3) = np.shape(data[0])
        
        inputdata = data[0].reshape((Nsize1*Nsize2*Nsize3), 1)/np.max(data[0])
        
        classif = GaussianMixture(n_components=self.classes, covariance_type="tied")
        classif.fit(inputdata)
        cluster = classif.predict(inputdata)
        segm = classif.means_[cluster]
        if (dimensdata == 2):
            segm = segm.reshape(Nsize1, Nsize3, Nsize2)
        else:
            segm = segm.reshape(Nsize1, Nsize2, Nsize3)
        maskGMM = segm.astype(np.float64) / np.max(segm)
        maskGMM = 255 * maskGMM # Now scale by 255
        maskGMM = maskGMM.astype(np.uint8) # obtain the GMM mask
            
        return [maskGMM]
    
    def nInput_datasets(self):
        return 1
    def nOutput_datasets(self):
        return 1