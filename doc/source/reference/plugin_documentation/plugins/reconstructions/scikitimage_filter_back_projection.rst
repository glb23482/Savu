Scikitimage Filter Back Projection
########################################################

Description
--------------------------

A Plugin to reconstruct an image by filter back projection using the inverse radon transform from scikit-image. 

Parameter definitions
--------------------------

.. code-block:: yaml

        in_datasets:
            visibility: datasets
            dtype: "[list[],list[str]]"
            description: 
                summary: A list of the dataset(s) to process.
                verbose: A list of strings, where each string gives the name of a dataset that was either specified by a loader plugin or created as output to a previous plugin.  The length of the list is the number of input datasets requested by the plugin.  If there is only one dataset and the list is left empty it will default to that dataset.
            default: "[]"
        
        out_datasets:
            visibility: datasets
            dtype: "[list[],list[str]]"
            description: 
                summary: A list of the dataset(s) to create.
                verbose: A list of strings, where each string is a name to be assigned to a dataset output by the plugin. If there is only one input dataset and one output dataset and the list is left empty, the output will take the name of the input dataset. The length of the list is the number of output datasets created by the plugin.
            default: "[]"
        
        centre_of_rotation:
            visibility: basic
            dtype: "[float, str, list[float], dict{int:float}]"
            description: Centre of rotation to use for the reconstruction.
            default: "0.0"
            example: It could be a scalar, a list, or a file containing centre of rotations
        
        init_vol:
            visibility: advanced
            dtype: "[None,str]"
            description: Dataset to use as volume initialiser (does not currently work with preview)
            default: None
            example: "Type the name of the initialised dataset e.g. ['tomo']"
        
        centre_pad:
            visibility: hidden
            dtype: "[bool,int,float]"
            description: Not required.
            default: "False"
        
        outer_pad:
            visibility: hidden
            dtype: "[bool,int,float]"
            description: Not required.
            default: "False"
        
        log:
            visibility: advanced
            dtype: bool
            description: 
                summary: Take the log of the data before reconstruction (true or false).
                verbose: Should be set to false if PaganinFilter is set beforehand
            default: "True"
            example: Set to True to take the log of the data before reconstruction
        
        preview:
            visibility: advanced
            dtype: preview
            description: A slice list of required frames.
            default: "[]"
            example: "[angle, detectorZ, detectorY], where detectorZ is the vertical coordinate, detectorY is the horizontal coordinate."
        
        force_zero:
            visibility: intermediate
            dtype: "[list[float,float],list[None,None]]"
            description: Set any values in the reconstructed image outside of this range to zero.
            default: "['None', 'None']"
            example: "[0,1]"
        
        ratio:
            visibility: intermediate
            dtype: float
            description: Ratio of the masks diameter in pixels to the smallest edge size along given axis.
            default: "0.95"
            example: "Is this a proper name for this parameter? Would mask_diameter or mask_circle be more accurate?"
        
        log_func:
            visibility: advanced
            dtype: str
            description: Override the default log function
            default: np.nan_to_num(-np.log(sino))
            example: You write a function as default
        
        vol_shape:
            visibility: basic
            dtype: "[str, int]"
            description: 
                summary: Override the size of the reconstruction volume with an integer value.
                verbose: When fixed, you get the dimension of the horizontal detector or you can specify any reconstruction size you like with an integer.
            default: fixed
        
        output_size:
            visibility: basic
            dtype: "[None, int,list[int,int],str]"
            description: Number of rows and columns in the reconstruction.
            default: auto
        
        filter:
            visibility: intermediate
            dtype: str
            description: Filter used in frequency domain filtering. Ramp filter used by default. Assign None to use no filter.
            options: "['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', 'None']"
            default: ramp
        
        interpolation:
            visibility: advanced
            dtype: int
            description: "Interpolation method used in reconstruction. Methods available: 'linear', 'nearest', and 'cubic' ('cubic' is slow)."
            options: "['linear', 'nearest', 'cubic']"
            default: linear
        
        circle:
            visibility: advanced
            dtype: bool
            description: Assume the reconstructed image is zero outside the inscribed circle. Also changes the default output_size to match the behaviour of radon called with circle=True.
            default: "False"
        
Key
^^^^^^^^^^

.. literalinclude:: /../source/files_and_images/plugin_guides/short_parameter_key.yaml
    :language: yaml

Citations
--------------------------

Principles Of Ct Imaging by Kak, Avinash C et al.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bibtex
""""""""""""""""""""""""""""""""""""""""""

.. code-block:: none

    @article{kak2002principles,
    title={Principles of computerized tomographic imaging},
    author={Kak, Avinash C and Slaney, Malcolm and Wang, Ge},
    journal={Medical Physics},
    volume={29},
    number={1},
    pages={107--107},
    year={2002},
    publisher={Wiley Online Library}
    }
    

Endnote
""""""""""""""""""""""""""""""""""""""""""

.. code-block:: none

    %0 Journal Article
    %T Principles of computerized tomographic imaging
    %A Kak, Avinash C
    %A Slaney, Malcolm
    %A Wang, Ge
    %J Medical Physics
    %V 29
    %N 1
    %P 107-107
    %@ 0094-2405
    %D 2002
    %I Wiley Online Library
    
