# Copyright 2015 Diamond Light Source Ltd.
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
.. module:: HDF5
   :platform: Unix
   :synopsis: Transport for saving and loading files using hdf5

.. moduleauthor:: Mark Basham <scientificsoftware@diamond.ac.uk>

"""

import logging
import numpy as np
import socket
import os
import copy

from mpi4py import MPI
from itertools import chain
from savu.core.utils import logfunction
from savu.data.transport_mechanism import TransportMechanism
from savu.core.utils import logmethod
from savu.data.data_structures import TomoRaw


class Hdf5Transport(TransportMechanism):

    def transport_control_setup(self, options):
        processes = options["process_names"].split(',')

        if len(processes) is 1:
            options["mpi"] = False
            options["process"] = 0
            options["processes"] = processes
            self.set_logger_single(options)
        else:
            options["mpi"] = True
            print("Options for mpi are")
            print(options)
            self.mpi_setup(options)

    def mpi_setup(self, options):
        print("Running mpi_setup")
        RANK_NAMES = options["process_names"].split(',')
        RANK = MPI.COMM_WORLD.rank
        SIZE = MPI.COMM_WORLD.size
        RANK_NAMES_SIZE = len(RANK_NAMES)
        if RANK_NAMES_SIZE > SIZE:
            RANK_NAMES_SIZE = SIZE
        MACHINES = SIZE/RANK_NAMES_SIZE
        MACHINE_RANK = RANK/MACHINES
        MACHINE_RANK_NAME = RANK_NAMES[MACHINE_RANK]
        MACHINE_NUMBER = RANK % MACHINES
        MACHINE_NUMBER_STRING = "%03i" % (MACHINE_NUMBER)
        ALL_PROCESSES = [[i]*MACHINES for i in RANK_NAMES]
        options["processes"] = list(chain.from_iterable(ALL_PROCESSES))
        options["process"] = RANK

        self.set_logger_parallel(MACHINE_NUMBER_STRING, MACHINE_RANK_NAME)

        MPI.COMM_WORLD.barrier()
        logging.info("Starting the reconstruction pipeline process")
        logging.debug("Rank : %i - Size : %i - host : %s", RANK, SIZE,
                      socket.gethostname())
        IP = socket.gethostbyname(socket.gethostname())
        logging.debug("ip address is : %s", IP)
        self.call_mpi_barrier()
        logging.debug("LD_LIBRARY_PATH is %s",  os.getenv('LD_LIBRARY_PATH'))
        self.call_mpi_barrier()

    @logfunction
    def call_mpi_barrier(self):
        logging.debug("Waiting at the barrier")
        MPI.COMM_WORLD.barrier()

    def set_logger_single(self, options):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(options["out_path"], 'log.txt'),
                                 mode='w')
        fh.setFormatter(logging.Formatter('L %(relativeCreated)12d M CPU0 0' +
                                          ' %(levelname)-6s %(message)s'))
        logger.addHandler(fh)
        logging.info("Starting the reconstruction pipeline process")

    def set_logger_parallel(self, number, rank):
        logging.basicConfig(level=0, format='L %(relativeCreated)12d M' +
                            number + ' ' + rank +
                            ' %(levelname)-6s %(message)s', datefmt='%H:%M:%S')
        logging.info("Starting the reconstruction pipeline process")

    def transport_run_plugin_list(self):
        """
        Runs a chain of plugins
        """
        exp = self.exp
        exp.barrier()
        logging.info("Starting the HDF5 plugin list runner")
        plugin_list = exp.meta_data.plugin_list.plugin_list

        exp.barrier()
        logging.info("run the loader plugin")
        self.plugin_loader(plugin_list[0])

        exp.barrier()
        logging.info("create all output data_objects and backing files")
        in_data = exp.index["in_data"][exp.index["in_data"].keys()[0]]
        out_data_objects = in_data.load_data(self)

        exp.barrier()
        logging.info("clear all out_data objects in experiment dictionary")
        exp.clear_data_objects()

        exp.barrier()
        logging.info("Load all the plugins")
        self.plugin_loader(plugin_list[0])

        exp.barrier()
        logging.info("Running all the plugins")

        for i in range(1, len(plugin_list)-1):
            logging.info("Running Plugin %s" % plugin_list[i]["id"])
            exp.barrier()

            logging.info("Initialise output data")
            for key in out_data_objects[i-1]:
                exp.index["out_data"][key] = out_data_objects[i-1][key]

            exp.barrier()
            logging.info("Load the plugin")
            plugin = self.plugin_loader(plugin_list[i], pos=i)

            exp.barrier()
            logging.info("run the plugin")
            return_dict = plugin.run_plugin(exp, self)

            try:
                remove_data_set = self.transfer_to_meta_data(
                    return_dict['transfer_to_meta_data'])
            except (KeyError, TypeError):
                remove_data_set = []
                pass

            exp.barrier()
            logging.info("Clean up input datasets")

            exp.barrier()
            logging.info("close any files that are no longer required")
            for out_objs in plugin.parameters["out_datasets"]:
                if out_objs in exp.index["in_data"].keys():
                    exp.index["in_data"][out_objs].save_data()
                elif out_objs in remove_data_set:
                    exp.index["out_data"][out_objs].save_data()
                    del exp.index["out_data"][out_objs]

            exp.barrier()
            logging.info("Copy out data to in data")
            for key in exp.index["out_data"]:
                exp.index["in_data"][key] = \
                    copy.deepcopy(exp.index["out_data"][key])

            exp.barrier()
            logging.info("Clear up all data objects")
            exp.clear_out_data_objects()

        exp.barrier()
        logging.info("close all remaining files")
        for key in exp.index["in_data"].keys():
            exp.index["in_data"][key].save_data()

        exp.barrier()
        logging.info("Completing the HDF5 plugin list runner")
        return

    @logmethod
    def timeseries_field_correction(self, plugin, in_data, out_data):

        expInfo = plugin.exp.meta_data
        in_data = in_data[0]
        out_data = out_data[0]

        in_slice_list, frame_list = in_data.data_obj.\
            get_slice_list_per_process(expInfo, frameList=True)
        out_slice_list, frame_list = out_data.data_obj.\
            get_slice_list_per_process(expInfo, frameList=True)

        for count in range(len(in_slice_list)):
            print count
            result = plugin.correction(in_data.data_obj.
                                       data[in_slice_list[count]],
                                       in_data.data_obj.get_image_key())
            out_data.data_obj.data[out_slice_list[count]] = result

    @logmethod
    def reconstruction_setup(self, plugin, in_data, out_data, expInfo):

        if isinstance(in_data, TomoRaw):
            raise Exception("The input data to a reconstruction plugin cannot \
            be Raw data. Have you performed a timeseries_field_correction?")

        [slice_list, frame_list] = \
            in_data.get_slice_list_per_process(expInfo, frameList=True)
        cor = in_data.meta_data.get_meta_data("centre_of_rotation")[frame_list]

        count = 0
        for sl in slice_list:
            frame = plugin.reconstruct(np.squeeze(in_data.data[sl]),
                                       cor[count],
                                       out_data.get_pattern_shape())
            out_data.data[sl] = frame
            count += 1
            plugin.count += 1
            logging.debug("Reconstruction progress (%i of %i)" %
                         (plugin.count, len(slice_list)))

    @logmethod
    def filter_chunk(self, plugin, in_data, out_data):
        logging.debug("Running filter._filter_chunk")

        expInfo = plugin.exp.meta_data
        in_slice_list = self.get_all_slice_lists(in_data, expInfo)
        out_slice_list = self.get_all_slice_lists(out_data, expInfo)

        for count in range(len(in_slice_list[0])):
            print count
            section = self.get_all_padded_data(in_data, in_slice_list, count)
            result = plugin.filter_frame(section)
            self.set_out_data(out_data, out_slice_list, result, count)
#
#    if type(result) == dict:
#        for key in result.keys():
#            if key == 'center_of_rotation':
#                frame = in_data[0].get_orthogonal_slice(in_slice_list[count],
#                in_data[0].core_directions[plugin.get_filter_frame_type()])
#                out_data.center_of_rotation[frame] = result[key]
#            elif key == 'data':
#                out_data.data[out_slice_list[count]] = \
#                in_data[0].get_unpadded_slice_data(in_slice_list[count],
#                                            padding, in_data[0], result)
#    else:
#        out_data.data[out_slice_list[count]] = \
#        in_data[0].get_unpadded_slice_data(in_slice_list[0][count], padding,
#                                        in_data[0], result)

    def get_all_slice_lists(self, data_list, expInfo):
        slice_list = []
        for data in data_list:
            slice_list.append(data.get_slice_list_per_process(expInfo))
        return slice_list

    def get_all_padded_data(self, data, slice_list, count):
        section = []
        for idx in range(len(data)):
            section.append(data[idx].get_padded_slice_data
                          (slice_list[idx][count]))
        return section

    def set_out_data(self, data, slice_list, result, count):
        result = [result] if type(result) is not list else result
        for idx in range(len(data)):
            data[idx].data[slice_list[idx][count]] = \
                data[idx].get_unpadded_slice_data(slice_list[idx][count],
                                                  result[idx])

    def transfer_to_meta_data(self, return_dict):
        remove_data_sets = []
        for data_key in return_dict.keys():
            for name in return_dict[data_key].keys():
                convert_data = return_dict[data_key][name]
                remove_data_sets.append(convert_data.name)
                data_key.meta_data.set_meta_data(name, convert_data.data[...])
        return remove_data_sets
