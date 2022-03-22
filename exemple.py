from skimage import data
import nibabel as ni
import napari

# viewer = napari.view_path('/home/nathan/PLEX/norm/sub-002/img.nii.gz')
viewer = napari.view_image(ni.load('/home/nathan/PLEX/norm/sub-002/img.nii.gz').get_fdata())

viewer.add_labels(ni.load('/home/nathan/PLEX/norm/sub-002/mask.nii.gz').get_fdata().astype('uint8'))
dw, my_widget = viewer.window.add_plugin_dock_widget('napari-labelprop', 'Inference')
my_widget.checkpoint.value='/home/nathan/ISO-DMD/bids2/checkpoints/bench/labelprop-epoch=08-val_accuracy=0.37-22032022-104701.ckpt'
napari.run()  # start the event loop and show viewer