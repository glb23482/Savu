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
.. module:: plugin
   :platform: Unix
   :synopsis: Base class for all plugins used by Savu

.. moduleauthor:: Mark Basham <scientificsoftware@diamond.ac.uk>

"""

import logging
import inspect

from savu.plugins import utils as pu
from savu.data.data_structures import PluginData


class Plugin(object):
    """
    The base class from which all plugins should inherit.
    """

    def __init__(self, name='Plugin'):
        super(Plugin, self).__init__()
        self.name = name
        self.exp = None
        self.parameters = {}
        self.data_objs = {}

    def main_setup(self, exp, params):
        self.exp = exp
        self.set_parameters(params)
        self.set_plugin_datasets(exp)
        self.setup()

    def setup(self):
        """
        This method is first to be called after the plugin has been created.

        :param in_data: Input data object(set to "None" if this is a loader)
        :type in_data: savu.data.experiment
        :param out_data: The output data object
        :type out_data: savu.data.experiment
        """
        logging.error("set_up needs to be implemented")
        raise NotImplementedError("setup needs to be implemented")

    def populate_default_parameters(self):
        """
        This method should populate all the required parameters with default
        values.  it is used for checking to see if new parameter values are
        appropriate

        It makes use of the classes including parameter information in the
        class docstring such as this
        :param error_threshold: Convergence threshold. Default: 0.001.
        """
        for clazz in inspect.getmro(self.__class__):
            if clazz != object:
                full_description = pu.find_args(clazz)
                for item in full_description:
                    self.parameters[item['name']] = item['default']

    def set_parameters(self, parameters):
        """
        This method is called after the plugin has been created by the
        pipeline framework

        :param parameters: A dictionary of the parameters for this plugin, or
            None if no customisation is required
        :type parameters: dict
        """
        self.parameters = {}
        self.populate_default_parameters()
        if parameters is not None:
            for key in parameters.keys():
                if key in self.parameters.keys():
                    self.parameters[key] = parameters[key]
                else:
                    raise ValueError("Parameter " + key +
                                     "is not a valid parameter for plugin " +
                                     self.name)

    def pre_process(self):
        """
        This method is called after the plugin has been created by the
        pipeline framework as a pre-processing step

        :param exp: An experiment object, holding input and output datasets
        :type exp: experiment class instance
        """
        pass

    def process(self, transport):
        """
        This method is called after the plugin has been created by the
        pipeline framework and forms the main processing step

        :param data: The input data object.
        :type data: savu.data.structures
        :param data: The output data object
        :type data: savu.data.structures
        :param processes: The number of processes which will be doing the work
        :type path: int
        :param path: The specific process which we are
        :type path: int
        """

        logging.error("process needs to be implemented")
        raise NotImplementedError("process needs to be implemented")

    def post_process(self):
        """
        This method is called after the process function in the pipeline
        framework as a post-processing step. All processes will have finished
        performing the main processing at this stage.

        :param exp: An experiment object, holding input and output datasets
        :type exp: experiment class instance
        """
        pass

    def clean_up(self):
        self.organise_metadata()
        self.clean_up_plugin_data()

    # Does this function have to be implemented: make default here that copies
    # the dictionary from the in data...
    def organise_metadata(self):
        """
        This method is called after the post_process function to organise the
        metadata that is passed from input datasets to output datasets

        :param exp: An experiment object, holding input and output datasets
        :type exp: experiment class instance
        """
        logging.error("organise_metadata() needs to be implemented")
        raise NotImplementedError("organise_metadata() needs to be "
                                  "implemented")

    def clean_up_plugin_data(self):
        in_data, out_data = self.get_datasets()
        data_object_list = in_data + out_data
        for data in data_object_list:
            data.clear_plugin_data()

    def nInput_datasets(self):
        """
        The number of datasets required as input to the plugin

        :returns:  Number of input datasets

        """
        raise NotImplementedError("nInput_datasets needs to be implemented")

    def nOutput_datasets(self):
        """
        The number of datasets created by the plugin

        :returns:  Number of output datasets

        """
        raise NotImplementedError("nOutput_datasets needs to be implemented")

    def get_citation_information(self):
        """Gets the Citation Information for a plugin

        :returns:  A populated savu.data.plugin_info.CitationInfomration

        """
        return None

    def get_data_objects(self, dtype):
        data_list = self.parameters[dtype + 'sets']
        data_objs = []
        for data in data_list:
            data_objs.append(self.exp.index[dtype][data])
        return data_objs

    def get_in_datasets(self):
        return self.get_data_objects('in_data')

    def get_out_datasets(self):
        try:
            out_data = self.get_data_objects('out_data')
        except KeyError:
            out_data = []
            for data in self.parameters['out_datasets']:
                self.exp.create_data_object("out_data", data)
            out_data = self.get_data_objects('out_data')
        return out_data

    def get_plugin_data(self, data_list):
        pattern_list = []
        for data in data_list:
            pattern_list.append(PluginData(data))
        return pattern_list

    def set_plugin_datasets(self, experiment):
        """
        Convert in/out_dataset strings to objects and create PluginData objects
        for each.
        """
        try:
            self.parameters['in_datasets'] = self.get_in_datasets()
            self.parameters['out_datasets'] = self.get_out_datasets()
            self.parameters['plugin_in_datasets'] = \
                self.get_plugin_data(self.parameters['in_datasets'])
            self.parameters['plugin_out_datasets'] = \
                self.get_plugin_data(self.parameters['out_datasets'])
        except KeyError:
            pass

    def get_plugin_datasets(self):
        plugin_in = self.parameters['plugin_in_datasets']
        plugin_out = self.parameters['plugin_out_datasets']
        return plugin_in, plugin_out

    def get_datasets(self):
        return self.parameters['in_datasets'], self.parameters['out_datasets']

    def get_meta_data(self):
        in_data, out_data = self.get_datasets()
        in_meta_data = self.set_meta_data(in_data, 'in_data')
        out_meta_data = self.set_meta_data(out_data, 'out_data')
        return in_meta_data, out_meta_data

    def set_meta_data(self, data_list, dtype):
        meta_data = []
        for data in data_list:
            meta_data.append(data.meta_data)
        return meta_data
