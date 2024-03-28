import datetime
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def k_cam_loss(preds, gts, return_mean=True, scores=None):
    """
    Calculate the K-CAM (Knowledge-Guided Class Activation Map) loss.

    Args:
        preds (list of torch.Tensor): List of predicted tensors.
        gts (list of torch.Tensor): List of ground truth tensors.
        return_mean (bool, optional): Whether to return the mean loss. Defaults to True.
        scores (list of torch.Tensor, optional): List of score tensors. If provided,
            element-wise multiplication with the difference between preds and gts is performed.
            Defaults to None.

    Returns:
        torch.Tensor: The K-CAM loss.

    Raises:
        ValueError: If lengths of preds and gts do not match, or if scores is not None
            and its length does not match preds and gts.
    """

    # Check if lengths of preds and gts match
    if len(preds) != len(gts):
        raise ValueError("Lengths of preds and gts must be the same.")

    # Check if lengths of scores match preds and gts (if scores is provided)
    if scores is not None and len(scores) != len(preds):
        raise ValueError("Length of scores must match preds and gts when scores is not None.")

    # Compute weights based on Frobenius norm of ground truth tensors
    weights = torch.stack([torch.linalg.norm(gt, ord="fro", axis=[-2, -1]) for gt in gts], dim=0).sum(dim=0).sum(dim=0)

    # Compute squared error
    if scores is None:
        squared_error = torch.stack(
            [torch.linalg.norm(pred - gt, ord="fro", axis=[-2, -1]) for pred, gt in zip(preds, gts)], dim=0).sum(
            dim=0).sum(dim=0)
    else:
        squared_error = torch.stack(
            [torch.linalg.norm(torch.mul((pred - gt), sc[:, :, None].repeat(1, 1, 2)), ord="fro", axis=[-2, -1]) for
             pred, gt, sc in zip(preds, gts, scores)], dim=0).sum(dim=0).sum(dim=0)
    # Calculate mean squared error if required
    if return_mean:
        mean_squared_error = (squared_error / weights).mean()
        return mean_squared_error
    else:
        return squared_error / weights


class TrainerMMPose(object):
    def __init__(self, train_data, net, device, cfg) -> None:
        """
        Initialize the TrainerMMPose.

        Args:
            train_data (torch.utils.data.Dataset): Training dataset.
            net (torch.nn.Module): The neural network model.
            device (torch.device): The device (CPU or GPU) on which to perform training.
            cfg (object): An object containing configuration parameters for the trainer.

        Attributes:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            loss_fn: The loss function used for training.
            optimizer (torch.optim.Optimizer): The optimizer for model training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            num_epochs (int): Number of training epochs.
            save_each_epoch (int): Interval for saving checkpoints.
            summary (object): Object containing summary of training parameters.
            exp_id (str): Experiment ID for naming checkpoints.
            device (torch.device): Selected device for training.
        """
        if train_data:
            self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=cfg.batch_size,
                                                            num_workers=cfg.num_workers)
        else:
            self.train_loader = None

        self.loss_fn = k_cam_loss
        self.net = net

        self.optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=1e-05)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.explr_gamma)

        self.num_epochs = cfg.num_epochs
        self.save_each_epoch = cfg.save_each_epoch

        self.summary = cfg  # Using the entire `cfg` object

        dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        exp_id = 'pannoptic_' + 'atoms' + str(self.summary.num_atoms) + '_bottleneck' + str(
            self.summary.num_atoms_bottleneck) + '_dic' + str(self.summary.num_dictionaries) + '_LR' + str(
            self.summary.learning_rate) + '_gamma' + str(self.summary.explr_gamma) + '_{}'
        exp_id = exp_id.format(dt)
        self.exp_id = exp_id
        self.device = device
        print(f'Selected device: {device}')

        if cfg.use_multigpu:
            self.net = nn.DataParallel(self.net)
            print("Training on Multiple GPUs")

        else:
            print("Training on a Single GPU or CPU")

    def prepare_checkpoint_folder(self, checkpoints_path='../checkpoints/'):
        """
        Prepare the checkpoint folder for saving model checkpoints.

        Returns:
            str: Path to the checkpoint folder.
        """
        checkpoint_folder = os.path.join(os.path.abspath(), self.exp_id)

        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        return checkpoint_folder

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            float: Mean loss for the epoch.
        """
        # Set train mode for both the encoder and the decoder
        self.net.train()
        total_loss = []

        # Iterate through the train_loader (unsupervised learning)
        for data in self.train_loader:
            measurements, _, scores = data

            # Move tensors to the proper device
            measurements = [measurement.to(self.device) for measurement in measurements]
            scores = [score.to(self.device) for score in scores]

            shapes, cameras = self.net(measurements)
            loss = self.loss_fn([shapes @ camera for camera in cameras], measurements, scores=scores)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Record batch loss
            total_loss.append(loss.item())

        self.scheduler.step()

        # Calculate and return the mean loss for the epoch
        mean_epoch_loss = np.mean(total_loss)
        return mean_epoch_loss

    def train(self, checkpoint_path=None):
        """
        Train the model for the specified number of epochs.

        Args:
            checkpoint_path (str, optional): Path to a checkpoint to resume training from.
        """
        log_path = os.path.join('../checkpoints/', self.exp_id)
        summary_writer = SummaryWriter(log_dir=log_path)
        if checkpoint_path is None:
            training_loss_history = {'train_loss': []}
            start_epoch = 0
        else:
            start_epoch, training_loss_history = self.load(checkpoint_path)

        for epoch in range(start_epoch, self.num_epochs):
            train_total_loss = self.train_epoch()

            training_loss_history['train_loss'].append(train_total_loss)
            summary_writer.add_scalar("Loss_Train/Total", train_total_loss, epoch)
            train_lr = self.scheduler.get_last_lr()[0]
            summary_writer.add_scalar("LR/Train", train_lr, epoch)

            if epoch % self.save_each_epoch == 0:
                self.save(epoch, training_loss_history)
                print('LR:', self.scheduler.get_last_lr())
            # Calculate validation loss if needed (add validation code here)

            logging.info(f'EPOCH {epoch + 1}/{self.num_epochs}\tTrain loss: {train_total_loss}')

        print('Finished Training')
        self.save(epoch, training_loss_history)

    def predict_mmpose(self):
        """
        Perform inference using the trained model on a dataset batch.

        Returns:
            tuple: A tuple containing two lists -
                1. List of dictionaries containing predictions for each data sample in the batch with keys:
                   - 'pred3d': Predicted 3D data.
                   - 'predCameras': Predicted camera parameters.
                   - 'predLosses': Loss values for the predictions.
                   - 'predProjs': Projected predictions.
                2. List of dictionaries containing input data for each sample in the batch with keys:
                   - 'points2d': 2D points.
                   - 'orig_points_2d': Original 2D points.
        """
        predictions = []
        inputs = []
        for data in self.train_loader:  # Ignore labels (second element of the train_loader tuple)

            proj_list, orig_list, score_list = data

            # Move tensor to the proper device
            measurements = [m.to(self.device) for m in proj_list]
            scores = [s.to(self.device) for s in score_list]

            pred3d, cameras = self.net(measurements)

            pred_projs = [pred3d @ camera for camera in cameras]

            with torch.no_grad():
                losses = [self.loss_fn([proj], [mea], return_mean=False) for proj, mea, scores in
                          zip(pred_projs, measurements, score_list)]

            pred3d = pred3d.detach().cpu().numpy()
            predCameras = np.asarray([camera.detach().cpu().numpy() for camera in cameras])
            predProjs = np.asarray([proj.detach().cpu().numpy() for proj in pred_projs])

            losses = np.asarray([loss.detach().cpu().numpy() for loss in losses])

            for i in range(len(data[0])):
                predCamera_i = predCameras[i]  # Use single indexing for 1D array
                predProj_i = predProjs[i]  # Use single indexing for 1D array
                loss_i = losses[i]  # Use single indexing for 1D array

                predictions.append(
                    dict(pred3d=pred3d[i], predCameras=predCamera_i, predLosses=loss_i, predProjs=predProj_i))

            proj = np.asarray([proj.numpy() for proj in proj_list])
            orig_proj = np.asarray([proj.numpy() for proj in orig_list])

            for i in range(len(data[0])):
                inputs.append(dict(points2d=proj[:, i], orig_points_2d=orig_proj[:, i]))

        return predictions, inputs

    def save(self, epoch, train_loss):
        """
        Save the model's current state, optimizer, and training-related information.

        Args:
            epoch (int): Current training epoch.
            train_loss (dict): Training loss history.

        Returns:
            str: Path to the saved checkpoint.
        """
        self.prepare_checkpoint_folder()
        checkpoint_path = os.path.join(self.checkpoint_folder, f'{epoch}.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'exp_id': self.exp_id,
            'loss': train_loss,
            'save_each_epoch': self.save_each_epoch,
            'num_epochs': self.num_epochs
        }, checkpoint_path)

        logging.info(f'Checkpoint saved: {checkpoint_path}')

        return checkpoint_path

    def load(self, checkpoint_path, load_all=True):
        """
        Load a saved checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            load_all (bool): If True, loads all training-related information. If False, loads only the model state.

        Returns:
            tuple or None: If load_all is True, returns (epoch, loss). If load_all is False, returns None.
        """
        checkpoint = torch.load(checkpoint_path)

        if load_all:
            self.exp_id = checkpoint['exp_id']
            self.save_each_epoch = checkpoint['save_each_epoch']
            self.num_epochs = checkpoint['num_epochs']
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            return epoch, loss
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
            return None

    def load_rem(self, checkpoint_path):
        """
        Load a saved checkpoint, restoring only the model and optimizer states.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
