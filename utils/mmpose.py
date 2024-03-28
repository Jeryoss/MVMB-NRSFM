import os
import numpy as np

from utils.h36m import PoseDatasetBase


class MMPose(PoseDatasetBase):
    def __init__(self, subjects):
        """
        Initialize the MMPose dataset.

        Args:
            subjects (list[str]): List of subject directories containing data files.
        """
        self.NUM_LANDMARKS = 17
        self.NUM_COORDS = 2
        self.CAMERA_COUNT = 3
        data_list = []
        camera_list = []
        for data_path in subjects:

            for index in range(self.CAMERA_COUNT):
                camera_list.append(np.load(os.path.join(data_path, f'c{index+1}.npy'), allow_pickle=True))
            for i, (p0, p1, p2) in enumerate(zip(*camera_list)):
                data_list.append(np.asarray([p0['landmarks'], p1['landmarks'], p2['landmarks']]))

        data_list = np.asarray(data_list)

        self.extended_data_list = data_list[:, :, :, :self.NUM_COORDS]
        self.score_2d = data_list[:, :, :, self.NUM_COORDS]

        # Normalize the 2D landmarks
        self._normalize_data()

    def _normalize_data(self):
        """Normalize the 2D landmarks and adjust the scores."""
        if self.extended_data_list.shape[-1] == self.NUM_COORDS:
            self.extended_data_list[:, :, :, -1] = (self.extended_data_list[:, :, :, -1] - 0.5) * 2

    def __len__(self):
        """Return the number of data samples in the dataset."""
        return len(self.extended_data_list)

    def __getitem__(self, i):
        """
        Get a single data sample from the dataset.

        Args:
            i (int): Index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing normalized landmarks, original landmarks, and scores.
        """
        _data = self.extended_data_list[i]
        _score = self.score_2d[i]

        norm_list = []
        orig_list = []
        score_list = []

        for d, s in zip(_data, _score):
            orig_list.append(d)

            dn = d.reshape(-1, self.NUM_LANDMARKS * self.NUM_COORDS)

            dn = self._normalize_2d(dn).astype(np.float32).reshape(self.NUM_LANDMARKS, self.NUM_COORDS)

            norm_list.append(dn)
            score_list.append(s)

        return norm_list, orig_list, score_list
