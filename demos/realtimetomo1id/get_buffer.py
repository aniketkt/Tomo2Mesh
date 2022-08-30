#import pyepics
import pvaccess as pva
import queue
import numpy as np
import time
from epics import PV

pv_name = "1idPG1:Pva1:Image"
pv_name2 = '1idPG1:cam1:AcquireTime'
nprojs = 10

exposure_time = PV('1idPG1:cam1:AcquirePeriod_RBV').get() # seconds
scan_time = exposure_time*nprojs + 100.0 # plus overhead just in case

data_queue = queue.Queue(maxsize=100)

def add_data(pv):

        #print(pv['value'][0]['ushortValue'].size/(1920*1200))
        print(PV('1idPG1:Pva1:UniqueId_RBV').get())

        if(data_queue.full()):
                print("Warning")

        else:
                # let's hope arr is a 2D numpy array
                arr = pv['value'][0]['ushortValue'] # (width,height)
                data_queue.put(arr)
        return


if __name__ == "__main__":

        pva_plugin_image = pva.Channel(pv_name)

        pva_plugin_image.subscribe("add_data", add_data)
        pva_plugin_image.startMonitor()

        #when to use stopMonitor() We are using the dumb way
        time.sleep(scan_time)
        pva_plugin_image.stopMonitor()
        print(PV('1idPG1:cam1:AcquireTime_RBV').get())
        # this is where we use data_queue.get()

        #projs = []
        #for ii in range(nprojs):
        #       projs.append(data_queue.get())

        #print(np.shape(projs))
        # hopefully this is (100, height, width)

        #theta = np.linspace(0,np.pi,nprojs, endpoint = True)
        #center = projs.shape[-1]/2.0 # just a guess

        # reconstruction, segmentation, void analysis code goes below
        #voids = some_func(projs, theta, center)
