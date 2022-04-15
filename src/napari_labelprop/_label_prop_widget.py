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
from magicgui.widgets import *
from napari.types import NewType
from labelprop.napari_entry import propagate_from_ckpt,train, train_and_infer
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
import numpy as np
import datetime
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
def inference(image: "napari.layers.Image", labels: "napari.layers.Labels", checkpoint: "napari.types.Path", z_axis: int, label : int) -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    shape=torch.load(checkpoint)['hyper_parameters']['shape'][0]
    if label==0: label='all'
    Y_up, Y_down, Y_fused = propagate_from_ckpt(
        image.data, labels.data, checkpoint, z_axis=z_axis,label=label,shape=shape)
    return [((Y_up).astype('uint8'), {'name': 'propagated_up','metadata':labels.metadata}, 'labels'), ((Y_down).astype('uint8'), {'name': 'propagated_down','metadata':labels.metadata}, 'labels'), ((Y_fused).astype('uint8'), {'name': 'propagated_fused','metadata':labels.metadata}, 'labels')]

#@magicgui(call_button='run')#(checkpoint_output_dir={'mode': 'd'}, call_button='Run') , checkpoint_output_dir: pathlib.Path.home()
@magic_factory(checkpoint_output_dir=dict(widget_type='FileEdit', mode='d'))
def training(image: "napari.layers.Image", labels: "napari.layers.Labels", pretrained_checkpoint: "napari.types.Path" = '/home/', shape: int=256, z_axis: int=0, max_epochs: int=10,checkpoint_output_dir = '/home/',checkpoint_name='',pretraining=False) -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    if not 'ckpt' in str(pretrained_checkpoint): pretrained_checkpoint=None
    else:
        shape=torch.load(pretrained_checkpoint)['hyper_parameters']['shape'][0]
    Y_up, Y_down, Y_fused = train_and_infer(
        image.data, labels.data, pretrained_checkpoint,shape,max_epochs,z_axis,str(checkpoint_output_dir),checkpoint_name,pretraining)
    torch.cuda.empty_cache()
    return [((Y_up).astype('uint8'), {'name': 'propagated_up','metadata':labels.metadata}, 'labels'), ((Y_down).astype('uint8'), {'name': 'propagated_down','metadata':labels.metadata}, 'labels'), ((Y_fused).astype('uint8'), {'name': 'propagated_fused','metadata':labels.metadata}, 'labels')]

def filter_slices(labels: "napari.layers.Labels",slices : str,z_axis: int=0) -> "napari.types.LayerDataTuple":
    slices=slices.replace(' ','').split(',')
    print(slices)
    labels_filtered=deepcopy(labels.data)
    indx = [slice(None)]*labels.ndim

    for i in range(labels.data.shape[z_axis]):
        if str(i) not in slices:
            indx[z_axis] = i
            labels_filtered[indx]=labels_filtered[indx]*0
    print(labels.metadata)
    return [((labels_filtered).astype('uint8'), {'name': 'filtered_mask','metadata':labels.metadata}, 'labels')]

def dice_coef(y_true, y_pred, smooth=1e-8):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def average_surface_distance(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    return np.mean(np.sqrt(np.sum((y_true_f - y_pred_f)**2, axis=1)))

def hausdorff_distance(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    return np.max(np.max(np.sqrt(np.sum((y_true_f - y_pred_f)**2, axis=1))))

def get_metrics(pred,gt):
    dice=dice_coef(pred,gt)
    hausdorff=hausdorff_distance(pred,gt)
    asd=average_surface_distance(pred,gt)
    return [((dice,hausdorff,asd), {'name': 'metrics'}, 'labels')]



# class GetMetrics(QWidget):
#     """
#     QWidget showing metrics between two LabelsData layers
#     """
#     def __init__(self,napari_viewer):
#         super().__init__()
#         self.viewer=napari_viewer
#         line_edit = LineEdit(value='hello!')
#         self.setLayout(QHBoxLayout())
#         print(napari_viewer.dict()['layers'])
#         date=LineEdit(bind=get_date())
#         self.layout().addWidget(line_edit.native)
#         self.layout().addWidget(date.native)
#         self.setWindowTitle('Metrics')
#         self.setMinimumWidth(300)
#         self.setMinimumHeight(300)
#         self.setLayout(QVBoxLayout())
#         self.layout().addWidget(QLabel('Dice Coefficient'))
#         self.dice_coef=QLabel('0')
#         self.layout().addWidget(self.dice_coef)
#         self.layout().addWidget(QLabel('Hausdorff Distance'))
#         self.hausdorff=QLabel('0')
#         self.layout().addWidget(self.hausdorff)
#         self.layout().addWidget(QLabel('Average Surface Distance'))
#         self.asd=QLabel('0')
#         #Button to show metrics
#         self.show_metrics_button=QPushButton('Show Metrics')
#         self.show_metrics_button.clicked.connect(self.show_metrics)
#         self.layout().addWidget(self.show_metrics_button)
#         #Add ListWidget showing available self.viewer.layers
#         self.layers_list=QListWidget(deepcopy(self.viewer.layers))
#         print(self.viewer.layers)
#         self.layout().addWidget(self.layers_list)

    
#     def show_metrics(self):
#         if len(self.viewer.layers)==2:
#             pred=self.viewer.layers[0].data
#             gt=self.viewer.layers[1].data
#             self.dice_coef.setText(str(dice_coef(pred,gt)))
#             self.hausdorff.setText(str(hausdorff_distance(pred,gt)))
#             self.asd.setText(str(average_surface_distance(pred,gt)))
#         else:
#             self.dice_coef.setText('0')
#             self.hausdorff.setText('0')
#             self.asd.setText('0')

@magic_factory(result_widget=True)
def GetMetrics(y_pred: "napari.layers.Labels",y_true: "napari.layers.Labels") -> QLabel :
    dice_score=dice_coef(y_pred.data,y_true.data)
    # hausdorff=hausdorff_distance(y_pred.data,y_true.data)
    # asd=average_surface_distance(y_pred.data,y_true.data)
    # print(dice,hausdorff,asd)
    return dice_score 

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