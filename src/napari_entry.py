
import train
from DataLoading import LabelPropDataModule
import numpy as np
import torch
from torch.nn import functional as func
from os.path import join
import shutil
import Pyro5

def resample(Y,size):
    Y=func.interpolate(Y[None,None,...]*1.,size,mode='nearest')[0,0]
    return Y


def propagate_from_ckpt(img,mask,checkpoint,shape=304,z_axis=2,lab='all'):

    true_shape=img.shape
    shape=(shape,shape)
    by_composition=True
    n_classes=int(np.max(mask))
    losses={'compo-reg-up':True,'compo-reg-down':True,'compo-dice-up':True,'compo-dice-down':True,'bidir-cons-reg':False,'bidir-cons-dice':False}
    model_PARAMS={'n_classes':n_classes,'way':'both','shape':shape,'selected_slices':None,'losses':losses,'by_composition':by_composition}
    print('hey ho')
    #Dataloading
    dm=LabelPropDataModule(img_path=img,mask_path=mask,lab=lab,shape=shape,selected_slices=None,z_axis=z_axis)

    #Inference
    Y_up,Y_down,Y_fused=train.inference(datamodule=dm,model_PARAMS=model_PARAMS,ckpt=checkpoint)
    if z_axis!=0:
        Y_up=torch.moveaxis(Y_up,0,z_axis)
        Y_down=torch.moveaxis(Y_down,0,z_axis)
        Y_fused=torch.moveaxis(Y_fused,0,z_axis)
    Y_up=resample(Y_up,true_shape)
    Y_down=resample(Y_down,true_shape)
    Y_fused=resample(Y_fused,true_shape)
    return Y_up.cpu().detach().numpy(),Y_down.cpu().detach().numpy(),Y_fused.cpu().detach().numpy()

def train_and_infer(img,mask,pretrained_ckpt,shape,max_epochs,z_axis=2,output_dir='~/label_prop_checkpoints',name='',pretraining=False):
    way='both'
    true_shape=img.shape
    shape=(shape,shape)
    by_composition=True
    n_classes=len(np.unique(mask))
    losses={'compo-reg-up':True,'compo-reg-down':True,'compo-dice-up':True,'compo-dice-down':True,'bidir-cons-reg':False,'bidir-cons-dice':False}
    model_PARAMS={'n_classes':n_classes,'way':way,'shape':shape,'selected_slices':None,'losses':losses,'by_composition':False}

    #Dataloading
    dm=LabelPropDataModule(img_path=img,mask_path=mask,lab='all',shape=shape,selected_slices=None,z_axis=z_axis)

    #Training and testing
    trained_model,best_ckpt=train.train(datamodule=dm,model_PARAMS=model_PARAMS,max_epochs=max_epochs,ckpt=pretrained_ckpt,pretraining=pretraining)
    best_ckpt=str(best_ckpt)
    Y_up,Y_down,Y_fused=train.inference(datamodule=dm,model_PARAMS=model_PARAMS,ckpt=best_ckpt)
    if z_axis!=0:
        Y_up=torch.moveaxis(Y_up,0,z_axis)
        Y_down=torch.moveaxis(Y_down,0,z_axis)
        Y_fused=torch.moveaxis(Y_fused,0,z_axis)
    Y_up=resample(Y_up,true_shape)
    Y_down=resample(Y_down,true_shape)
    Y_fused=resample(Y_fused,true_shape)

    if name=='': name=best_ckpt.split('/')[-1]

    shutil.copyfile(best_ckpt,join(output_dir,f'{name.split(".ckpt")[-1]}.ckpt'))
    return Y_up.cpu().detach().numpy(),Y_down.cpu().detach().numpy(),Y_fused.cpu().detach().numpy()

def train_and_infer_pyro(shape,max_epochs,z_axis=2,output_dir='~/label_prop_checkpoints',name='',pretraining=False):
    img=receive_object_from_client('img')
    mask=receive_object_from_client('mask')
    pretrained_ckpt=receive_object_from_client('pretrained_ckpt')
    Y_up,Y_down,Y_fused=train_and_infer(img,mask,pretrained_ckpt,shape,max_epochs,z_axis,output_dir,name,pretraining)
    send_object_to_server(Y_up,'Y_up')
    send_object_to_server(Y_down,'Y_down')
    send_object_to_server(Y_fused,'Y_fused')
    

def receive_object_from_client(name):
    with Pyro5.locate_ns() as ns:
        uri = ns.lookup(name)
    with Pyro5.Proxy(uri) as proxy:
        return proxy

def send_object_to_server(obj,name):
    with Pyro5.locate_ns() as ns:
        with Pyro5.Proxy(ns.lookup('example.server')) as proxy:
            proxy.register(obj,name)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='train':
        img=receive_object_from_client('img')
        mask=receive_object_from_client('mask')
        pretrained_ckpt=receive_object_from_client('pretrained_ckpt')
        shape=int(sys.argv[2])
        max_epochs=int(sys.argv[3])
        z_axis=int(sys.argv[4])
        output_dir=sys.argv[5]
        name=sys.argv[6]
        pretraining=bool(sys.argv[7])
        train_and_infer(img,mask,pretrained_ckpt,shape,max_epochs,z_axis,output_dir,name,pretraining)
    elif sys.argv[1]=='infer':
        shape=int(sys.argv[2])
        max_epochs=int(sys.argv[3])
        z_axis=int(sys.argv[4])
        output_dir=sys.argv[5]
        name=sys.argv[6]
        pretraining=bool(sys.argv[7])
        train_and_infer_pyro(shape,max_epochs,z_axis,output_dir,name,pretraining)
    else:
        raise ValueError('Invalid argument')