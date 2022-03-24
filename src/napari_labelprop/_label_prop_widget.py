"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""

from magicgui import magic_factory,magicgui
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton,QVBoxLayout,QLabel,QFileDialog,QListWidget,QLineEdit,QListWidgetItem
from qtpy.QtCore import Signal, QObject, QEvent
from qtpy.QtCore import QEvent, Qt
from napari.types import NewType
from napari_entry import propagate_from_ckpt,train, train_and_infer
import sys
import pathlib
from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode
from os import listdir
from os.path import isfile, join
import fnmatch
from magicgui.widgets import create_widget
import torch
from copy import deepcopy
sys.path.append('../MiniLabelProp')

class MyQLineEdit(QLineEdit):
    keyup = Signal()
    keydown = Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.keyup.emit()
            return
        elif event.key() == Qt.Key_Down:
            self.keydown.emit()
            return
        super().keyPressEvent(event)

class FolderBrowser(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())


        # --------------------------------------------
        # Directory selection
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        self.layout().addWidget(QLabel("Directory"))
        filename_edit = FileEdit(
            mode=FileDialogMode.EXISTING_DIRECTORY,
            value=file)
        self.layout().addWidget(filename_edit.native)

        def directory_changed(*args, **kwargs):
            self.current_directory = str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/")
            self.all_files = [f for f in listdir(self.current_directory) if isfile(join(self.current_directory, f))]

            text_changed() # update shown list

        filename_edit.line_edit.changed.connect(directory_changed)

        # --------------------------------------------
        #  File filter
        self.layout().addWidget(QLabel("File filter"))
        seach_field = MyQLineEdit("*")
        results = QListWidget()

        # update search
        def text_changed(*args, **kwargs):
            search_string = seach_field.text()

            results.clear()
            for file_name in self.all_files:
                if fnmatch.fnmatch(file_name, search_string):
                    _add_result(results, file_name)
            results.sortItems()

        # navigation in the list
        def key_up():
            if results.currentRow() > 0:
                results.setCurrentRow(results.currentRow() - 1)

        def key_down():
            if results.currentRow() < results.count() - 1:
                results.setCurrentRow(results.currentRow() + 1)

        seach_field.keyup.connect(key_up)
        seach_field.keydown.connect(key_down)
        seach_field.textChanged.connect(text_changed)

        # open file on ENTER and double click
        def item_double_clicked():
            item = results.currentItem()
            print("opening", item.file_name)
            self.viewer.open(join(self.current_directory, item.file_name))

        seach_field.returnPressed.connect(item_double_clicked)
        #results.itemDoubleClicked.connect(item_double_clicked)
        results.itemActivated.connect(item_double_clicked)

        self.setLayout(QVBoxLayout())

        w = QWidget()
        w.setLayout(QHBoxLayout())
        w.layout().addWidget(QLabel("Search:"))
        w.layout().addWidget(seach_field)
        self.layout().addWidget(w)

        self.layout().addWidget(results)

        directory_changed() # run once to initialize

def _add_result(results, file_name):
    item = QListWidgetItem(file_name)
    item.file_name = file_name
    results.addItem(item)

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
        filename_edit = FileEdit(
            mode=FileDialogMode.EXISTING_DIRECTORY,
            value='/')
        self.layout().addWidget(filename_edit.native)
        self.layout().addWidget(training.native)
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

#@magicgui(call_button='run')#(checkpoint_output_dir={'mode': 'd'}, call_button='Run') , checkpoint_output_dir: pathlib.Path.home()
@magic_factory(checkpoint_output_dir=dict(widget_type='FileEdit', mode='d'))
def training(image: "napari.types.ImageData", labels: "napari.types.LabelsData", pretrained_checkpoint: "napari.types.Path" = '/home/', shape: int=256, z_axis: int=0, max_epochs: int=10,checkpoint_output_dir = '/home/',checkpoint_name='') -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    if not 'ckpt' in str(pretrained_checkpoint): pretrained_checkpoint=None
    else:
        shape=torch.load(pretrained_checkpoint)['hyper_parameters']['shape'][0]
    Y_up, Y_down, Y_fused = train_and_infer(
        image, labels, pretrained_checkpoint,shape,max_epochs,z_axis,str(checkpoint_output_dir),checkpoint_name)
    torch.cuda.empty_cache()
    return [((Y_up).astype(int), {'name': 'propagated_up'}, 'labels'), ((Y_down).astype(int), {'name': 'propagated_down'}, 'labels'), ((Y_fused).astype(int), {'name': 'propagated_fused'}, 'labels')]

def filter_slices(labels: "napari.types.LabelsData",slices : str,z_axis: int=0) -> "napari.types.LabelsData":
    slices=slices.replace(' ','').split(',')
    print(slices)
    labels_filtered=deepcopy(labels)
    indx = [slice(None)]*labels.ndim

    for i in range(labels.shape[z_axis]):
        if str(i) not in slices:
            indx[z_axis] = i
            labels_filtered[indx]=labels_filtered[indx]*0
    return labels_filtered


class FuseLabelWidget(QWidget):
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
        filename_edit = FileEdit(
            mode=FileDialogMode.EXISTING_DIRECTORY,
            value='/')
        self.layout().addWidget(filename_edit.native)
        self.layout().addWidget(training.native)
    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")