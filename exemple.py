from skimage import data
import nibabel as ni
import napari


# viewer = napari.view_path('/home/nathan/PLEX/norm/sub-002/img.nii.gz')
viewer = napari.view_image(ni.load('/mnt/freebox/Segmentations/sub-17/ses-01/anat/sub-17_ses-1_DIXON6ECHOS-e3.nii.gz').get_fdata())

viewer.add_labels(ni.load('/mnt/freebox/Segmentations/sub-17/ses-01/anat/seg.nii.gz').get_fdata().astype('uint8'))
dw, my_widget = viewer.window.add_plugin_dock_widget('napari-labelprop', 'Training')
my_widget.checkpoint_output_dir.value='/home/nathan/checkpoints/'
my_widget.checkpoint_name.value='pretraining3'
my_widget.z_axis.value=2
my_widget.pretraining.value=True
napari.run()

