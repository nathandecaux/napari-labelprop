# import napari
# import numpy as np
# from matplotlib.backends.backend_qt5agg import FigureCanvas
# from matplotlib.figure import Figure
# import bokeh

# import sys
# #importing Widgtes
# from PyQt5.QtWidgets import *
# #importing Engine Widgets
# from PyQt5.QtWebEngineWidgets import QWebEngineView
# #importing QtCore to use Qurl
# from PyQt5.QtCore import *


# with napari.gui_qt():
#     mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
#     static_ax = mpl_widget.figure.subplots()
#     t = np.linspace(0, 10, 501)
#     static_ax.plot(t, np.tan(t), ".")
#     browser = QWebEngineView()
#     #setting url for browser, you can use any other url also
#     browser.setUrl(QUrl('http://google.com'))
#     viewer = napari.Viewer()
#     viewer.window.add_dock_widget(browser)

from skimage import data
import nibabel as ni
import napari


# viewer = napari.view_path('/home/nathan/PLEX/norm/sub-002/img.nii.gz')
viewer = napari.view_image(ni.load('/home/nathan/Datasets/PLEX/bids/norm/sub-005/img.nii.gz').get_fdata())

viewer.add_labels(ni.load('/home/nathan/Datasets/PLEX/bids/norm/sub-005/mask.nii.gz').get_fdata().astype('uint8'))
dw, my_widget = viewer.window.add_plugin_dock_widget('napari-labelprop', 'Training')
my_widget.checkpoint_output_dir.value='/home/nathan/checkpoints/'
my_widget.checkpoint_name.value='pretraining3'
my_widget.z_axis.value=2
my_widget.pretraining.value=True
napari.run()