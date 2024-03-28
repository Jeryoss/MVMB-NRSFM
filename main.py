import datetime
import os.path
import sys

import numpy as np
import torch

sys.path.append('.')
sys.path.append('..')

from model import Multi_RF_Nrsfm_Net
from trainer import Trainer
from utils.panoptic import Panoptic
from utils.misc import Struct
from utils.h36m import PoseDatasetBase

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class PanopticPipeline:
    landmarks_connections = np.array(
        [[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7],
         [7, 8], [8, 9], [3, 13], [13, 14], [14, 15], [1, 10],
         [10, 11], [11, 12]]) - 1

    colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
    arrow_edges = np.array([[0, 19], [0, 20], [0, 21]])

    def __init__(self, data_path="/nas/database/panoptic/", train_subjects=None,
                 test_subjects=None,
                 pose_estimator="gt", input_type="video",
                 number_of_landmarks=19, people_count=-1,
                 checkpoint_path='./notebooks/checkpoints/last_checkpoint.pt',
                 cameras=None, use_cache=True, cache_path='data/panoptic_custom_data_band',
                 num_atoms=1024, num_atoms_bottleneck=16,
                 num_dictionaries=7, exp_atom_decrease=True,
                 BLOCK_HEIGHT=3, BLOCK_WIDTH=2,
                 num_points=19, batch_size=64,
                 num_workers=8, n_cams=3,
                 learning_rate=0.0001, explr_gamma=0.95,
                 num_epochs=600, use_multigpu=True, cuda_id=0, save_each_epoch=10,
                 load_checkpoint=False):

        # fix random seed for torch to 0
        torch.manual_seed(0)
        # Main parameters
        self.train_subjects = train_subjects
        self.test_subjects = test_subjects
        self.data_path = data_path
        self.pose_estimator = pose_estimator
        self.input_type = input_type
        self.number_of_landmarks = number_of_landmarks
        self.camera_count = n_cams
        self.people_count = people_count
        self.pose_data = PoseDatasetBase()

        # Output related parameters
        self.experiment_id = self.get_experiment_id()
        self.output_visualization = f"output/bbox_test_small_{self.experiment_id}"

        # Input related parameters
        self.checkpoint_path = checkpoint_path
        self.cfg = self.set_config(num_atoms=num_atoms, num_atoms_bottleneck=num_atoms_bottleneck,
                                   num_dictionaries=num_dictionaries, exp_atom_decrease=exp_atom_decrease,
                                   BLOCK_HEIGHT=BLOCK_HEIGHT, BLOCK_WIDTH=BLOCK_WIDTH,
                                   num_points=num_points, batch_size=batch_size,
                                   num_workers=num_workers, n_cams=n_cams,
                                   learning_rate=learning_rate, explr_gamma=explr_gamma,
                                   num_epochs=num_epochs, use_multigpu=use_multigpu, cuda_id=cuda_id,
                                   save_each_epoch=save_each_epoch)

        self.device = self.initialize_cuda()

        self.test_data = self.load_data(self.data_path, self.test_subjects, cameras=cameras, train=False,
                                        use_cache=use_cache,
                                        cache_path=f"{cache_path}/")

        self.train_data = self.load_data(self.data_path, self.train_subjects, cameras=cameras, train=True,
                                         use_cache=use_cache,
                                         cache_path=f"{cache_path}/")

        # self.test_data = self.train_data
        # Load the model
        self.net = Multi_RF_Nrsfm_Net(self.cfg)
        self.trainer = Trainer(self.train_data, self.test_data, self.net, self.device, self.cfg)
        if load_checkpoint:
            self.trainer.load(self.checkpoint_path)

    def load_data(self, data_path, subjects, cameras, train=True, use_cache=True,
                  cache_path='panoptic_custom_data_band/'):

        return Panoptic(data_path=data_path, train=train, n_cams=self.camera_count, subjects=subjects, cameras=cameras,
                        use_cache=use_cache,
                        cache_path=cache_path)

    def train(self, checkpoint_path=None):
        self.trainer.train(checkpoint_path)

    def set_config(self, num_atoms=2 ** 11, num_atoms_bottleneck=16,
                   num_dictionaries=7, exp_atom_decrease=True,
                   BLOCK_HEIGHT=3, BLOCK_WIDTH=2,
                   num_points=19, batch_size=128,
                   num_workers=8, n_cams=3,
                   learning_rate=0.001, explr_gamma=0.997,
                   num_epochs=600, use_multigpu=True, cuda_id=0, save_each_epoch=10):

        num_atoms_bottleneck = num_atoms // (2 ** (num_dictionaries - 1))

        cfg = dict(num_atoms=num_atoms,
                   num_atoms_bottleneck=num_atoms_bottleneck,
                   num_dictionaries=num_dictionaries,
                   exp_atom_decrease=exp_atom_decrease,
                   BLOCK_HEIGHT=BLOCK_HEIGHT,
                   BLOCK_WIDTH=BLOCK_WIDTH,
                   num_points=num_points,
                   batch_size=batch_size,
                   num_workers=num_workers,
                   n_cams=n_cams,
                   learning_rate=learning_rate,
                   explr_gamma=explr_gamma,
                   num_epochs=num_epochs,
                   use_multigpu=True,
                   cuda_id=0,
                   exp_id=self.experiment_id,
                   step_size=2,
                   save_each_epoch=10)
        cfg = Struct(**cfg)

        return cfg

    def get_experiment_id(self):

        """
        Returns a unique experiment ID.

        Returns:
            A unique experiment ID.
        """

        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{self.pose_estimator}_{self.camera_count}_{self.number_of_landmarks}_{timestamp}"

    def initialize_cuda(self):

        """
        Initializes CUDA.

        Returns:
            The CUDA device.
        """

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        if self.cfg.use_multigpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda:%d" % self.cfg.cuda_id if torch.cuda.is_available() else "cpu")
        gpu_brand = torch.cuda.get_device_name(self.cfg.cuda_id) if use_cuda else None
        gpu_count = torch.cuda.device_count() if self.cfg.use_multigpu else 1
        if use_cuda:
            print('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))
        return device

    def evaluate(self):
        self.dataframe, self.predicted_data = self.trainer.eval()
        print(self.dataframe.describe())


from panoptic_utils import calculate_error_rate, get_connected_samples, train_video, calculate_video_error

# Example usage of the PoseEstimationPipeline class
if __name__ == "__main__":
    train_subjects = [
        '160422_ultimatum1',
        '160224_haggling1',
        '160226_haggling1',
        '161202_haggling1',
        '160906_ian1',
        '160906_ian2',
        '160906_ian3',
        '160906_band1',
        '160906_band2',
        # '160906_band3',
    ]

    test_subjects = [
        '160906_pizza1', '160422_haggling1', '160906_ian5',
        '160906_band4'
    ]
    train_subjects = test_subjects
    cameras = np.array([3, 12, 23])  #
    pipeline = PanopticPipeline(train_subjects=test_subjects, test_subjects=test_subjects, cameras=cameras, n_cams=3,
                                use_cache=True, num_epochs=50, batch_size=512,
                                load_checkpoint = True,
                                checkpoint_path="checkpoints/3_cameras_fixed.pt")
    # pipeline.train()

    # MV NRSFM evaluation
    pipeline.evaluate()



    # MVMB evaluation
    error = calculate_error_rate(pipeline.predicted_data)
    connected_samples_Frames = get_connected_samples(pipeline.predicted_data)
    train_video(connected_samples_Frames, pipeline.predicted_data, pipeline.cfg)
