# napari-labelprop

[![License](https://img.shields.io/pypi/l/napari-labelprop.svg?color=green)](https://github.com/nathandecaux/napari-labelprop/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-labelprop.svg?color=green)](https://pypi.org/project/napari-labelprop)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-labelprop.svg?color=green)](https://python.org)
[![tests](https://github.com/nathandecaux/napari-labelprop/workflows/tests/badge.svg)](https://github.com/nathandecaux/napari-labelprop/actions)
[![codecov](https://codecov.io/gh/nathandecaux/napari-labelprop/branch/main/graph/badge.svg)](https://codecov.io/gh/nathandecaux/napari-labelprop)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-labelprop)](https://napari-hub.org/plugins/napari-labelprop)

Label propagation through deep registration.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

To install this project :

    pip install napari['all']
    git clone https://github.com/nathandecaux/napari-labelprop.git
    cd napari-labelprop
    pip install -e .


## Usage

Open napari from terminal and start using functions from 'napari-labelprop' plugin (Under Plugins scrolling menu). 

Available functions are :
- Inference : Propagate labels from trained weights (Pytorch checkpoint required)
- Training : Start training from scratch or from a pretrained model
- Remove annotated slices : (testing purpose) Function to remove every annotations except for declared slices. Kept slices must be declared in the 'slices' field using comma (',') separation (eg. 5,12,43)

PS : "pretraining" option in the Training menu is still under development

Alternatively, napari and plugin widgets can be called directly from python scripts : 

```python
import nibabel as ni
import napari

viewer = napari.view_image(ni.load('images.nii.gz').get_fdata())
viewer.add_labels(ni.load('segmentation.nii.gz').get_fdata().astype('uint8'))
dw, my_widget = viewer.window.add_plugin_dock_widget('napari-labelprop', 'Training')
my_widget.checkpoint_output_dir.value='~'
my_widget.checkpoint_name.value='checkpoint_name'
my_widget.z_axis.value=2
my_widget.pretraining.value=False
napari.run()
```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-labelprop" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
