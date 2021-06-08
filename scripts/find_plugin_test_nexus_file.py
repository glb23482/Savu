#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 09:20:34 2021

@author: qmm55171
"""
# find which test plugin lists contain a plugin class
# find the test module that call these plugin lists

import argparse
import h5py
import os

from pathlib import Path

def __option_parser(doc=True):
    """ Option parser for command line arguments. Use -d for file deletion
    and -q for quick template.
    """
    parser = argparse.ArgumentParser(prog='Find test nexus file containing specified plugin')
    parser.add_argument('plugin',
                        help='plugin class name',
                        type=str)

    return parser.parse_args()

def _find_test_modules(tests, nxs_file):
    for t in tests:
        with open(t) as f:
            if os.path.basename(nxs_file) in f.read():
                print("\t--->", t)

def main():
    base_folder = "/home/algol/Documents/DEV/Savu/test_data/"
    args = __option_parser(doc=False)
    plugin_name = args.plugin
    process_lists = list(Path(base_folder + "test_process_lists").rglob("*.nxs"))
    process_lists += list(Path(base_folder + "process_lists").rglob("*.nxs"))

    test_path = "/home/algol/Documents/DEV/Savu/savu/test/"
    tests = list(Path(test_path + "travis").rglob("*.py"))
    tests += list(Path(test_path + "jenkins").rglob("*.py"))

    for nxs_file in process_lists:
        try:
            with h5py.File(nxs_file, 'r') as f:
                #breakpoint()
                for k in list(f['entry/plugin'].keys()):
                    name = (f['entry/plugin'][k]['name'][...])[0].decode('utf-8')
                    if plugin_name == name:
                        print(nxs_file)
                        _find_test_modules(tests, nxs_file)
        except OSError:
            pass


if __name__ == '__main__':
    main()
