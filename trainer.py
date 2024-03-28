import os
import sys
import numpy as np
import datetime


import torch
from torch import nn

from utils.motion_capture import get_figure3d
from utils.reconstruction import evaluate_reconstruction
from torch.utils.tensorboard import SummaryWriter

from utils.h36m import H36M

### Define the loss function
def k_cam_loss(preds, gts):
    
    weights = torch.stack([torch.linalg.norm(gt, ord="fro", axis=[-2, -1]) for gt in gts], dim=0).sum(dim=0).sum(dim=0)
    squared_error = torch.stack([torch.linalg.norm(pred - gt, ord="fro", axis=[-2, -1]) for pred, gt in zip(preds, gts)], dim=0).sum(dim=0).sum(dim=0)
    mean_squared_error = (squared_error / weights).mean()
    
    return mean_squared_error


class Trainer(object):
    def __init__(self, train_data, test_data, net, device, cfg) -> None:

        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        print("Train loader length:", len(self.train_loader))
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        print("Test loader length:", len(self.test_loader))

        self.loss_fn = k_cam_loss
        self.net = net

        self.optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=1e-05)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.explr_gamma)

        self.num_epochs = cfg.num_epochs
        self.save_each_epoch = cfg.save_each_epoch

        self.summary = {"batch_size":cfg.batch_size, "num_atoms":cfg.num_atoms, "num_atoms_bottleneck":cfg.num_atoms_bottleneck,
         "num_dictionaries":cfg.num_dictionaries, "BLOCK_HEIGHT":cfg.BLOCK_HEIGHT, "BLOCK_WIDTH":cfg.BLOCK_WIDTH, "num_points":cfg.num_points,
          "num_workers":cfg.num_workers, "n_cams":cfg.n_cams, "learning_rate":cfg.learning_rate, "explr_gamma":cfg.explr_gamma }

        dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        exp_id = 'pannoptic_' + 'atoms' + str(self.summary['num_atoms']) +'_bottleneck'+ str(self.summary['num_atoms_bottleneck']) +'_dic' +str(self.summary['num_dictionaries'])+'_LR'+str(self.summary['learning_rate']) +'_gamma'+ str(self.summary['explr_gamma'])+'_{}'
        exp_id = exp_id.format(dt)
        self.exp_id = exp_id
        self.device = device
        print(f'Selected device: {device}')

        # Move both the encoder and the decoder to the selected device
        self.net.to(device)

        if cfg.use_multigpu:
            self.net = nn.DataParallel(self.net)
            print("Training on Multiple GPU's")

    def prepare_checkpoint_folder(self):
        self.checpoint_folder = os.path.join('./checkpoints/', self.exp_id)
        
        if not os.path.exists(self.checpoint_folder):        
            os.makedirs(self.checpoint_folder)

    ### Training function
    def train_epoch(self):
        # Set train mode for both the encoder and the decoder
        self.net.train()
        train_loss = []
        # Iterate the train_loader (we do not need the label values, this is unsupervised learning)
        for data in self.train_loader:

            if type(self.train_loader.dataset) is H36M:
                measurements, _, _ = data
            else:
                measurements, _, _, _, _, _, _, _ = data

            # Move tensor to the proper device
            measurements = [measurement.to(self.device) for measurement in measurements]
                    
            shapes, cameras = self.net(measurements)
            # Evaluate loss

            
            loss = self.loss_fn([shapes @ camera for camera in cameras], measurements)
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Print batch loss
            #print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        self.scheduler.step()

        return np.mean(train_loss)
    ### Validation function
    def val_epoch(self):
        # Set val mode for both the encoder and the decoder
        self.net.eval()
        val_loss = []
        # Iterate the train_loader (we do not need the label values, this is unsupervised learning)
        for data in self.test_loader:

            if type(self.test_loader.dataset) is H36M:
                measurements, _, _ = data
            else:
                measurements, _, _, _, _, _, _, _ = data
            # Move tensor to the proper device
            measurements = [measurement.to(self.device) for measurement in measurements]
                    
            shapes, cameras = self.net(measurements)
            # Evaluate loss
            with torch.no_grad():
                loss = self.loss_fn([shapes @ camera for camera in cameras], measurements)
      
                val_loss.append(loss.detach().cpu().numpy())

        self.scheduler.step()
        return np.mean(val_loss) 



    def train(self, checpoint_path=None):
        
        log_path=os.path.join('./checkpoints/', self.exp_id)
        summaryWriter = SummaryWriter(log_dir=log_path)
        if checpoint_path is None:
            diz_loss = {'train_loss':[]}
            start_epoch = 0

        else:
            start_epoch, diz_loss = self.load(checpoint_path)

        
        for epoch in range(start_epoch, self.num_epochs):
            train_loss = self.train_epoch()
            diz_loss['train_loss'].append(train_loss)
            summaryWriter.add_scalar("Loss/Train", train_loss, epoch)
            train_lr=self.scheduler.get_last_lr()[0]
            summaryWriter.add_scalar("LR/Train", train_lr,epoch)

            if epoch % self.save_each_epoch == 0:
                self.save(epoch, diz_loss)
                print('LR:', self.scheduler.get_last_lr())
            #caculate validation loss
            
            val_loss = self.val_epoch()
            
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, self.num_epochs,train_loss,val_loss))
           
            summaryWriter.add_scalar("Loss/Validation", val_loss, epoch)
            
        print('Finished Training')
        self.save(epoch, diz_loss)

        #evaluate the metrics
        dataframe, data=self.eval()
        df_error_2d=dataframe['error_2d'].describe().to_dict()
        df_error_3d=dataframe['error_3d'].describe().to_dict()
        df_error_abs=dataframe['error_abs'].describe().to_dict()
        summaryWriter.add_hparams({
                                  'batch_size':self.summary['batch_size'], 'num_epochs':self.num_epochs, 'num_atoms':self.summary['num_atoms'],
                                  'num_atoms_bottleneck':self.summary['num_atoms_bottleneck'],'num_dictionaries':self.summary['num_dictionaries'],
                                  'BLOCK_HEIGHT':self.summary['BLOCK_HEIGHT'],'BLOCK_WIDTH':self.summary['BLOCK_WIDTH'],'num_points':self.summary['num_points'],
                                   'num_workers':self.summary['num_workers'],'n_cams':self.summary['n_cams'],'learning_rate':self.summary['learning_rate'],
                                   'explr_gamma':self.summary['explr_gamma']
                                  },
                                  {
                                   'hparam/error_2d_count':df_error_2d['count'],'hparam/error_2d_mean':df_error_2d['mean'],
                                    'hparam/error_3d_count':df_error_3d['count'],'hparam/error_3d_mean':df_error_3d['mean'],
                                    'hparam/error_abs_count':df_error_abs['count'],'hparam/error_abs_mean':df_error_abs['mean'],

                                   'hparam/error_2d_std':df_error_2d['std'],'hparam/error_2d_min':df_error_2d['min'],
                                   'hparam/error_2d_max':df_error_2d['max'],'hparam/error_2d_25p':df_error_2d['25%'],
                                   'hparam/error_2d_50p':df_error_2d['50%'],'hparam/error_2d_75p':df_error_2d['75%'],
                                   
                                  
                                   'hparam/error_3d_std':df_error_3d['std'],'hparam/error_3d_min':df_error_3d['min'],
                                   'hparam/error_3d_max':df_error_3d['max'],'hparam/error_3d_25p':df_error_3d['25%'],
                                   'hparam/error_3d_50p':df_error_3d['50%'],'hparam/error_3d_75p':df_error_3d['75%'],
                                   
                                   
                                  
                                  'hparam/error_abs_std':df_error_abs['std'],'hparam/error_abs_min':df_error_abs['min'],
                                  'hparam/error_abs_max':df_error_abs['max'],'hparam/error_abs_25p':df_error_abs['25%'],
                                  'hparam/error_abs_50p':df_error_abs['50%'],'hparam/error_abs_75p':df_error_abs['75%']

                                  })


    
    def predict(self):
        predictions = []
        inputs = []
        for data in self.test_loader: # with "_" we just ignore the labels (the second element of the train_loader tuple)
            
            if type(self.test_loader.dataset) is H36M:
                proj_list, points3d_list, scale_list = data
            else:
                proj_list, orig_proj_list, points3d_list, scale_list, seq_names, frame_indeces, body_indeces, cam_list = data

            # Move tensor to the proper device
            measurements = [measurement.to(self.device) for measurement in proj_list]

            pred3d, cameras = self.net(measurements)

            
            
            pred3d=pred3d.detach().cpu().numpy()

            predCameras = np.asarray([camera.detach().cpu().numpy() for camera in cameras])
            #predUs = np.asarray([u.detach().cpu().numpy() for _, u, _ in cameras])
            #predVs = np.asarray([v.detach().cpu().numpy() for _, _, v in cameras])

            for i in range(len(pred3d)):
                predictions.append(dict(pred3d=pred3d[i], predCameras=predCameras[:, i]))


            proj = np.asarray([proj.numpy() for proj in proj_list])
            orig_proj = np.asarray([proj.numpy() for proj in orig_proj_list])
            points3d = np.asarray([points3d.numpy() for points3d in points3d_list])

            scale_list = np.asarray([scale.numpy() for scale in scale_list])
            
            gt_cameras = [[self.test_loader.dataset.seq_cameras[seq][cam_id] for cam_id in cam_ids] for seq, cam_ids in zip(seq_names, cam_list)]

            frame_indeces = [int(frame_idx.numpy()) for frame_idx in frame_indeces]
            poses = [self.test_loader.dataset.seq_skels[seq_id][frame_idx][body_idx].reshape(-1, 4)[:, :3] for seq_id, frame_idx, body_idx in zip(seq_names, frame_indeces, body_indeces)]
            
            
            for i in range(len(seq_names)):

                

                inputs.append(dict(points3d=points3d[:, i],
                                    points2d= proj[:, i],
                                    origpoints2d=orig_proj[:, i],
                                    predCameras=predCameras[:, i],
                                    seq_name=seq_names[i],
                                    frame_idx=frame_indeces[i],
                                    body_idx=body_indeces[i],
                                    gt_cameras=gt_cameras[i],
                                    scales = scale_list[:, i],
                                    pose=poses[i]))

        return predictions, inputs

    def eval(self):
        predictions, inputs = self.predict()        

        return evaluate_reconstruction(predictions, inputs)

        


    def save(self, epoch, train_loss):   

        self.prepare_checkpoint_folder()     
            
        checpoint_path = os.path.join(self.checpoint_folder, '{}.pt'.format(epoch))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'exp_id': self.exp_id,
            'loss': train_loss,
            'save_each_epoch': self.save_each_epoch,
            'num_epochs': self.num_epochs
            }, checpoint_path)

        print('Checkpoint saved:', checpoint_path)

        return checpoint_path

    def load(self, checpoint_path):
        checkpoint = torch.load(checpoint_path)
        self.exp_id = checkpoint['exp_id']
        self.save_each_epoch = checkpoint['save_each_epoch'] 
        self.num_epochs = checkpoint['num_epochs'] 
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return epoch, loss

    def load_rem(self, checpoint_path):
        checkpoint = torch.load(checpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        