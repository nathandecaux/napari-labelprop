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
from psygnal import Signal, throttled, debounced

from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode
from os import listdir
from os.path import isfile, join
import fnmatch
from magicgui.widgets import create_widget
import torch
import time
from contextlib import redirect_stdout
import io
from torch.nn.functional import one_hot
from copy import deepcopy
import numpy as np
import datetime
import napari
from skimage import morphology
from skimage.segmentation import slic
import napari.qt.threading as nqt
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
# a widget

def inference_function(image: "napari.layers.Image", labels_layer: "napari.layers.Labels", hints: "napari.layers.Labels",  checkpoint: "napari.types.Path", shape=(256,256), z_axis: int=0, label_n : int=0,criteria='ncc',reduction='none',gpu=True) -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    # shape=torch.load(checkpoint)['hyper_parameters']['shape'][0]
    device='cuda' if gpu else 'cpu'
    kwargs={'criteria':criteria,'reduction':reduction,'device':device}
    checkpoint=str(checkpoint)
    labels_data=labels_layer.data.astype('uint8')
    if hints==labels_layer:
        hints=''

    if hints!='':
        hints_data=hints.data.astype('uint8')
    else:
        hints_data=None
    if label_n>0:
        labels_data=(labels_data==label_n)*1
        if hints!='':
            hints_data=(hints_data==label_n)*1
    if label_n==0: label_n='all'
    Y_up, Y_down, Y_fused = propagate_from_ckpt(
        image.data, labels_data, checkpoint, hints=hints_data, z_axis=z_axis,label=label_n,shape=shape,**kwargs)
    
    return (Y_fused, {"name":"Propagated","affine": labels_layer.affine, "metadata": labels_layer.metadata}, "labels")
    # return [((Y_up).astype('uint8'), {'name': 'propagated_up','metadata':labels.metadata}, 'labels'), ((Y_down).astype('uint8'), {'name': 'propagated_down','metadata':labels.metadata}, 'labels'), ((Y_fused).astype('uint8'), {'name': 'propagated_fused','metadata':labels.metadata}, 'labels')]

class inference(FunctionGui):
    def __init__(self,viewer: "napari.viewer.Viewer"):         
        super().__init__(inference_function,call_button=True,param_options={'criteria':{'choices':['distance','ncc']},'reduction':{'choices':['none','local_mean','mean']},'checkpoint':{'filter':'*.ckpt'}})
        #Change display name of parameters
        self.criteria.changed.connect(self.update_reduction)
        self.image.label='Image'
        self.labels_layer.label='Labels'
        self.hints.label='(Optional) Additional Scribbles'
        self.shape.label='Slices shape'
        self.checkpoint.label='Checkpoint'
        self.z_axis.label='Propagation axis'
        self.label_n.label='Label to propagate (0 for all)'
        self.criteria.label='Weighting criteria'
        self.reduction.label='Reduction'
        self.gpu.label='Use GPU'

        #Disable shape second dimension
        self.shape[1].enabled=False
        if self.image.value is not None:
            self.update_shape()
        #Add signal to update shape when image is changed (with a wait of 1s)
        self.shape[0].changed.connect(self.update_second_dim)

        #Change the possible increments of the shape
        self.shape[0].step=16

        self.image.changed.connect(self.update_shape)
        self.z_axis.changed.connect(self.update_shape)

    def __call__(self):
        napari.utils.notifications.show_info('Inference started')
        #Call super in an unblocking thread (avoid WARNING QObject::setParent: Cannot set parent, new parent is in a different thread)
        worker=self._call_super()
    #     worker.returned.connect(self._on_finished)
    #     worker.start()
    # # @nqt.thread_worker(connect={'returned':_on_finished})
    # @nqt.thread_worker
    def _call_super(self):
        super().__call__()
    
    def _on_finished(self, result):
        napari.utils.notifications.show_info('Inference finished')
        
        
    def update_reduction(self):
        if self.criteria.value=='distance':
            self.reduction.value='mean'
            self.reduction.hide()
        else:
            self.reduction.show()

    def update_shape(self):
        if self.image.value is not None:
            img_shape=self.image.value.data.shape[:self.z_axis.value]+self.image.value.data.shape[self.z_axis.value+1:]
            self.shape.value=tuple(img_shape)
            self.shape[0].value=16*round(self.shape[0].value/16)
            self.shape[1].value=16*round(self.shape[1].value/16)
            self.img_shape=tuple(img_shape)
        
    def update_first_dim(self):
        if self.image.value is not None:
            
            ratio=self.img_shape[0]/self.img_shape[1]
            self.shape[0].value=int(ratio*self.shape[1].value)
    
    def update_second_dim(self):
        if self.shape[0].value is not None and self.shape[0].value>10:
            #Round first dim to the closest number divisible by 16
            self.shape[0].value=16*round(self.shape[0].value/16)
            if self.image.value is not None:
                print(self.img_shape)
                ratio=self.img_shape[1]/self.img_shape[0]
                new_val=int(ratio*self.shape[0].value)
                #Round to the closest number divisible by 8
                self.shape[1].value=16*round(new_val/16)


#@magicgui(call_button='run')#(checkpoint_output_dir={'mode': 'd'}, call_button='Run') , checkpoint_output_dir: pathlib.Path.home()
# @magic_factory(checkpoint_output_dir=dict(widget_type='FileEdit', mode='d'))
def training_function(image: "napari.layers.Image", labels_layer: "napari.layers.Labels",hints: "napari.layers.Labels", pretrained_checkpoint: "napari.types.Path", shape=(256,256), z_axis: int=0,label_n: int=0, max_epochs: int=50,checkpoint_output_dir = '/tmp/checkpoints/',checkpoint_name='',criteria='ncc',reduction='none',gpu=True) -> "napari.types.LayerDataTuple":
    """Generate thresholded image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    pretrained_checkpoint=None if not 'ckpt' in str(pretrained_checkpoint) else str(pretrained_checkpoint)
    print(pretrained_checkpoint)
    device='cuda' if gpu else 'cpu'
    kwargs={'criteria':criteria,'reduction':reduction,'device':device}
    labels_data=labels_layer.data.astype('uint8')

    if hints==labels_layer:
        hints=''

    if hints!='':
        hints_data=hints.data.astype('uint8')
    else:
        hints_data=None

    if label_n>0:
        labels_data=(labels_data==label_n)*1
        if hints!='':
            hints_data=(hints_data==label_n)*1
    if label_n==0: label_n='all'
    print(checkpoint_name)
    Y_up, Y_down, Y_fused = train_and_infer(
        image.data, labels_data, pretrained_checkpoint,shape,max_epochs,z_axis,str(checkpoint_output_dir),checkpoint_name,hints=hints_data,pretraining=False,**kwargs)
    torch.cuda.empty_cache()
    napari.utils.notifications.show_info('Training finished')

    return (Y_fused, {"name":"Propagated","affine": labels_layer.affine, "metadata": labels_layer.metadata}, "labels")



class training(FunctionGui):
    def __init__(self,viewer: "napari.viewer.Viewer"):
        super().__init__(training_function,call_button=True,param_options={'criteria':{'choices':['distance','ncc']},'reduction':{'choices':['none','local_mean','mean']}, 'checkpoint_output_dir':{'widget_type':'FileEdit','mode': 'd'},'pretrained_checkpoint':{'filter':'*.ckpt'}})
        self.criteria.changed.connect(self.update_reduction)
        self.image.label='Image'
        self.labels_layer.label='Labels'
        self.hints.label='(Optional) Additional Scribbles'
        self.pretrained_checkpoint.label='Pretrained checkpoint'
        self.shape.label='Slices shape'
        self.z_axis.label='Propagation axis'
        self.label_n.label='Label to propagate (0 for all)'
        self.max_epochs.label='Max epochs'
        self.checkpoint_output_dir.label='Checkpoint output directory'
        #Get current directory
        self.checkpoint_output_dir.value=str(pathlib.Path.cwd())
        self.checkpoint_name.label='Checkpoint name'
        self.criteria.label='Weighting criteria'
        self.reduction.label='Reduction'
        self.gpu.label='Use GPU'

        #Disable shape second dimension

        if self.image.value is not None:
            self.update_shape()
        #Add signal to update shape when image is changed (with a wait of 1s)

        self.shape[0].changed.connect(self.update_second_dim)
        #Change the possible increments of the shape
        self.shape[0].step=16
        self.shape[1].enabled=False
        # self.shape[1].changed.connect(self.update_first_dim)
        self.image.changed.connect(self.update_shape)
        self.z_axis.changed.connect(self.update_shape)
        self.call_button.clicked.connect(self._on_click)
    
    def _on_click(self):
        napari.utils.notifications.show_info('Training started')

    def update_reduction(self):
        if self.criteria.value=='distance':
            self.reduction.value='mean'
            self.reduction.hide()
        else:
            self.reduction.show()

    def update_shape(self):
        if self.image.value is not None:
            img_shape=self.image.value.data.shape[:self.z_axis.value]+self.image.value.data.shape[self.z_axis.value+1:]
            self.shape.value=tuple(img_shape)
            self.shape[0].value=16*round(self.shape[0].value/16)
            self.shape[1].value=16*round(self.shape[1].value/16)
            self.img_shape=tuple(img_shape)
        
    def update_first_dim(self):
        if self.image.value is not None:
            
            ratio=self.img_shape[0]/self.img_shape[1]
            self.shape[0].value=int(ratio*self.shape[1].value)
    
    def update_second_dim(self):
        if self.shape[0].value is not None and self.shape[0].value>10:
            #Round first dim to the closest number divisible by 16
            self.shape[0].value=16*round(self.shape[0].value/16)
            if self.image.value is not None:
                print(self.img_shape)
                ratio=self.img_shape[1]/self.img_shape[0]
                new_val=int(ratio*self.shape[0].value)
                #Round to the closest number divisible by 8
                self.shape[1].value=16*round(new_val/16)

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

def get_supervoxels(image: "napari.layers.Image",n_segments : int=100,compactness: float=0.1,mask_threshold:float=0,slic_zero: bool=False) -> "napari.types.LayerDataTuple":
    """Generate supervoxels. Based on slic function from scikit-image.

    This function will be turned into a widget using `autogenerate: true`.
    """
    mask=(image.data-np.min(image.data))/(np.max(image.data)-np.min(image.data))
    mask=mask>mask_threshold
    supervoxels=slic(image.data, multichannel=False, n_segments=n_segments, compactness=compactness,slic_zero=slic_zero,mask=mask)
    return [((supervoxels).astype('uint8'), {'name': 'supervoxels','metadata':image.metadata}, 'labels')]

def dice_coef(y_true, y_pred, smooth=1e-8):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

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
def GetMetrics(y_pred: "napari.layers.Labels",y_true: "napari.layers.Labels",z_axis=0) -> QLabel :
    print(y_pred.data.shape)
    pred=torch.from_numpy(y_pred.data).long()
    gt=torch.from_numpy(y_true.data).long()
    pred_oh=torch.moveaxis(one_hot(pred,pred.max()+1),-1,0)
    pred_oh=torch.moveaxis(pred_oh, z_axis+1, 0)
    y_true_oh=torch.moveaxis(one_hot(gt,gt.max()+1),-1,0)
    y_true_oh=torch.moveaxis(y_true_oh, z_axis+1, 0)
    dices={}
    print(dices)
    for lab in list(range(y_true_oh.shape[1]))[1:]:
        dices[lab]=[]
        for i in range(y_true_oh.shape[0]):
            if y_true_oh[i,lab].sum()>0:
                dices[lab].append(dice_coef(pred_oh[i,lab],y_true_oh[i,lab]))
    # hausdorff=hausdorff_distance(y_pred.data,y_true.data)
    # asd=average_surface_distance(y_pred.data,y_true.data)
    # print(dice,hausdorff,asd)
    for k,v in dices.items():
        dices[k]=torch.stack(v).mean()
    return str(dices) 

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

def remove_small_objects(labels:"napari.layers.Labels",min_size:int=64,connectivity:int=1,z_axis:int=2) -> "napari.types.LayerDataTuple":
    # filter=labels.data*False
    # for lab in list(np.unique(labels.data))[1:]:
    #     label=labels.data==lab
    #     mask=morphology.remove_small_objects(label,min_size=min_size,connectivity=connectivity)
    #     filter=filter+mask
    filtered=one_hot(torch.from_numpy(labels.data.copy().astype('uint8')).long())>0
    for lab in list(np.unique(labels.data))[1:]:
        for i in range(labels.data.shape[z_axis]):
            label=filtered[:,:,i,lab].numpy()
            mask=morphology.remove_small_objects(label,min_size=min_size,connectivity=connectivity)
            filtered[:,:,i,lab]=torch.from_numpy(mask)
    filtered=torch.argmax(filtered*1.,dim=-1).numpy()
    return [((filtered).astype('uint8'), {'name': 'filtered_mask','metadata':labels.metadata}, 'labels')]