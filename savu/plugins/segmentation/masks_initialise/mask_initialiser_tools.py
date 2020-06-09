from savu.plugins.plugin_tools import PluginTools

class MaskInitialiserTools(PluginTools):
    """A plugin to initialise a binary mask for level sets and \
distance transform segmentations. Seeds are generated by \
providing coordinates of three points in 3D space (start-middle-finish) \
and connecting them with a cylinder of a certain radius. \
Importantly the Z coordinate is given following VOLUME_XY vertical pattern
    """
    def define_parameters(self):
        """
        mask1_coordinates:
            visibility: basic
            dtype: list
            description: "X0,Y0,Z0 (start) X1,Y1,Z1 (middle) and \
              X2,Y2,Z2 (finish) coordinates of three points."
            default: [10, 10, 0, 15, 15, 15, 20, 20, 20]

        mask1_radius:
            visibility: basic
            dtype: int
            description: Mask1 will be initialised with an ellipse of radius.
            default: 5

        mask2_coordinates:
            visibility: basic
            dtype: list
            description: The second mask coordinates.
            default: None

        mask2_radius:
            visibility: basic
            dtype: int
            description: Mask2 will be initialised with an ellipse of radius.
            default: None

        out_datasets:
            visibility: datasets
            dtype: list
            description: The default names
            default: ['INIT_MASK']

        """