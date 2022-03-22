"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""

from magicgui import magic_factory
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from napari_entry import propagate_from_ckpt,train, train_and_infer
import sys
sys.path.append('../MiniLabelProp')


class TrucQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


@magic_factory
def magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer.data.shape}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def inference(image: "napari.types.ImageData", labels: "napari.types.LabelsData", checkpoint: "napari.types.Path", z_axis: int, label : int) -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    print(labels.shape)
    if label==0: label='all'
    Y_up, Y_down, Y_fused = propagate_from_ckpt(
        image, labels, checkpoint, z_axis=z_axis,lab=label)
    return [((Y_up).astype(int), {'name': 'propagated_up'}, 'labels'), ((Y_down).astype(int), {'name': 'propagated_down'}, 'labels'), ((Y_fused).astype(int), {'name': 'propagated_fused'}, 'labels')]


def training(image: "napari.types.ImageData", labels: "napari.types.LabelsData", pretrained_checkpoint: "napari.types.Path", z_axis: int, max_epochs: int, checkpoint_output_dir: "napari.types.Path") -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    shape=256
    Y_up, Y_down, Y_fused = train_and_infer(
        image, labels, pretrained_checkpoint,shape,max_epochs,z_axis,checkpoint_output_dir)
    return [((Y_up).astype(int), {'name': 'propagated_up'}, 'labels'), ((Y_down).astype(int), {'name': 'propagated_down'}, 'labels'), ((Y_fused).astype(int), {'name': 'propagated_fused'}, 'labels')]
