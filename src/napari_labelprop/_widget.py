"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton,QLabel
from qtpy.QtGui import QPixmap
from magicgui import magic_factory,magicgui
import napari

class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer = viewer

        btn = QPushButton("Click me!")
        
        self.setLayout(QHBoxLayout())
        # self.layout().addWidget(btn)
        label = QLabel('Bonjour')
        pixmap = QPixmap('test.jpg')
        label.setPixmap(pixmap)
        label2 = QLabel(self)
        label2.setText(str('Truc'))
        self.layout().addWidget(label)
        label.show()
    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
    
    


def example_magic_widget():
    logo='test.jpg'
    @magicgui(label_head=dict(widget_type='Label', label=f'<h1 style="text-align:center"><img src="{logo}"></h1>'))
    def widget(viewer: napari.viewer.Viewer,label_head):
        print('bonjour')

# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")

