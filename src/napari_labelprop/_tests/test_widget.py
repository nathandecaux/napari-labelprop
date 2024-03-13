from napari_labelprop import ExampleQWidget, example_magic_widget
import numpy as np

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
    
def test_example_magic_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # this time, our widget will be a MagicFactory or FunctionGui instance
    dw, my_widget = viewer.window.add_plugin_dock_widget('napari-labelprop', 'Training')
    dw2, my_widget2 = viewer.window.add_plugin_dock_widget('napari-labelprop', 'Inference')
    

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
