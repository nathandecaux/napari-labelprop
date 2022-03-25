import Pyro5
import Pyro5.api
import nibabel as ni

@Pyro5.api.expose
class GreetingMaker(object):
    def __init__(self, volume):
        self.volume = volume
    def get_shape(self):
        return self.volume.shape

vol=ni.load("/mnt/freebox/Segmentations/sub-09/ses-01/anat/sub-9_ses-1_T2.nii.gz").get_fdata()
daemon = Pyro5.server.Daemon()         # make a Pyro daemon
ns = Pyro5.api.locate_ns()             # find the name server
uri = daemon.register(GreetingMaker(vol))   # register the greeting maker as a Pyro object
ns.register("example.greeting", uri)   # register the object with a name in the name server

print("Ready.")
daemon.requestLoop()                   # start the event loop of the server to wait for calls