from savu.plugins.plugin_tools import PluginTools
from savu.plugins.utils import register_plugin_tool

@register_plugin_tool
class NxtomoLoaderTools(PluginTools):
    """A class to load tomography data from a hdf5 file
    """
    def define_parameters(self):
        """---
        preview:
            visibility: basic
            dtype: '[int]'
            description: A slice list of required frames.
            default: []
        name:
            visibility: basic
            dtype: str
            description: A name assigned to the dataset.
            default: 'tomo'
        data_path:
            visibility: basic
            dtype: str
            description: Path to the data inside the file.
            default: null
        image_key_path:
            visibility: basic
            dtype: int_path
            description: Path to the image key entry inside the nxs file.
              Set this parameter to None if use this loader for radiography.
            default: 'entry1/tomo_entry/instrument/detector/image_key'
        dark:
            visibility: basic
            dtype: '[path, int_path, int]'
            description: Specify the nexus file location where the dark field
              images are stored. Then specify the path within this nexus file,
              at which the dark images are located. The last value will be a
              scale value.
            default: '[None, None, 1]'
        flat:
            visibility: basic
            dtype: '[path, int_path, int]'
            description: This parameter needs to be specified only if flats
              not stored in the same dataset as sample projections. Optional
              Path to the flat field data file, nxs path and scale value.
            default: '[None, None, 1]'
        angles:
            visibility: basic
            dtype: '[int]'
            description: If this if 4D data stored in 3D then pass an integer
              value equivalent to the number of projections per 180 degree
              scan. If the angles parameter is set to None, then values from
              default dataset used.
            default: 'None'
        three_to_4d:
            visibility: basic
            dtype: bool
            options: ['False', 'True']
            description:
              summary: Many tomography datasets can be loaded. Value of
                True indicates the data must be reshaped.
            default: 'False'
        ignore_flats:
            visibility: basic
            dtype: '[int]'
            description: List of batch numbers of flats to ignore (starting
              at 1). Useful for excluding comprimised flats in combined data
              sets containing multiple batches of interspaced flats. The
              batch indexing begins at 1.
            default: 'None'

        """