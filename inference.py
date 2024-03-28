import datetime
import os
import sys
import time

import cv2
import numpy as np
import plotly.offline as py
import torch

sys.path.append('.')
sys.path.append('..')

from mmpose_utils.model import Multi_RF_Nrsfm_Net
from mmpose_utils.trainer_mmpose import TrainerMMPose
from utils.misc import Struct
from utils.generate_video_from_frames import create_video
from utils.motion_capture import export_frames_as_images
from utils.h36m import PoseDatasetBase

from mmpose_utils.file_writer import load_json

from mmpose_utils.draw_bb import draw_bounding_boxes
from mmpose_utils.similarity import compute_similarity

from itertools import product
from mmpose_utils.reprojection import reprojection_loss
from scipy.optimize import minimize

import threading

# add os cuda visiable devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class PoseEstimationPipeline:
    def __init__(self):

        self.subjects = ['data/panoptic/']
        self.pose_estimator = "rtmpose"
        self.input_type = "video"

        self.data = None
        self.trainer = None

        self.number_of_landmarks = 17
        self.camera_count = 2
        self.people_count = 2  # leave -1 to set the count as dynamic otherwise specify the number of people

        self.pose_data = PoseDatasetBase()
        self.landmarks_connections = np.array(
            [[0, 3], [0, 4], [3, 5], [5, 7], [7, 9], [3, 11], [11, 13], [13, 15], [4, 6], [6, 8], [8, 10], [4, 12],
             [12, 14],
             [14, 16], [11, 12]])

        self.experiment_id = self.get_experiment_id()
        self.fine_tuned_path = f"fine_tuned_models/{self.experiment_id}"
        self.output_visualization = f"output/bbox_test_small_{self.experiment_id}"

        # self.checkpoint_path = 'checkpoints/pannoptic_atoms1024_bottleneck16_dic7_LR0.001_gamma0.997_20230412154529/599.pt'
        self.checkpoint_path = 'checkpoints/3_cameras_random.pt'
        self.cfg = self.set_config()
        self.device = self.initialize_cuda()

    def set_config(self, num_atoms=2024, num_atoms_bottleneck=32,
                   num_dictionaries=7, exp_atom_decrease=True,
                   BLOCK_HEIGHT=3, BLOCK_WIDTH=2,
                   num_points=17, batch_size=32,
                   num_workers=8, n_cams=3,
                   learning_rate=0.001, explr_gamma=0.997,
                   num_epochs=600, use_multigpu=True, cuda_id=0, save_each_epoch=10):

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
                   use_multigpu=use_multigpu,
                   cuda_id=cuda_id,
                   save_each_epoch=save_each_epoch,
                   exp_id=self.experiment_id)

        num_atoms = 2 ** 11  # 4096

        num_dictionaries = 7
        num_atoms_bottleneck = num_atoms // (2 ** (num_dictionaries - 1))
        # in_features = cfg.num_atoms_bottleneck * cfg.BLOCK_HEIGHT * cfg.BLOCK_WIDTH
        # out_features = cfg.BLOCK_HEIGHT * cfg.BLOCK_WIDTH + cfg.num_atoms_bottleneck
        # TODO fix block height and width
        cfg = dict(num_atoms=num_atoms,
                   num_atoms_bottleneck=num_atoms_bottleneck,
                   num_dictionaries=num_dictionaries,
                   exp_atom_decrease=True,
                   BLOCK_HEIGHT=3,
                   BLOCK_WIDTH=2,
                   num_points=17,
                   batch_size=64,
                   num_workers=8,
                   n_cams=3,
                   learning_rate=0.0005,
                   explr_gamma=0.95,
                   num_epochs=100,
                   use_multigpu=True,
                   cuda_id=0,
                   exp_id=self.experiment_id,
                   step_size=2,
                   save_each_epoch=10)
        # cfg = Struct(**cfg)

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

    def load_network(self):

        """
        Loads the network.
        """

        net = Multi_RF_Nrsfm_Net(self.cfg)
        trainer = TrainerMMPose(self.data, net, self.device, self.cfg)
        trainer.load(self.checkpoint_path, load_all=False)
        self.trainer = trainer

    def save_3d_predictions(self, predictions):

        """
        Saves the 3D predictions.

        Args:
            predictions: The predictions.
        """

        frames_ = [np.array([prediction["pred3d"]]) for prediction in predictions]
        export_frames_as_images(frames_, output_folder="output_folder", range_scale=3,
                                connections=[self.landmarks_connections])
        create_video("output_folder", "out.mp4", 30)

    def visualize_prediction(self, predictions, index=200):

        """
        Visualizes the prediction.

        Args:
            predictions: The predictions.
            index: The index of the prediction to visualize.
        """

        fig = self.get_figure3d([predictions[index]["pred3d"]], range_scale=3, connections=[self.landmarks_connections])
        py.iplot(fig)

    def calibrate_by_procrustes(self, points3d, camera, gt):

        """
        Performs calibration by Procrustes.

        Args:
            points3d: The predicted 3D points.
            camera: The camera matrix.
            gt: The ground truth 3D points.

        Returns:
            The rotation matrix.
        """

        if camera is not None:
            singular_value = np.linalg.norm(camera, 2)
            camera = camera / singular_value
            points3d = points3d * singular_value
        scale = np.linalg.norm(gt) / np.linalg.norm(points3d)
        points3d = points3d * scale
        U, s, Vh = np.linalg.svd(points3d.T.dot(gt))
        rot = U.dot(Vh)
        return rot

    def k_cam_loss(self, predictions, gt_labels, return_mean=True, scores=None):

        """
        Computes the K-CAM loss.

        Args:
            predictions: The predicted camera matrices.
            gt_labels: The ground truth camera matrices.
            return_mean: Whether to return the mean loss.
            scores: The scores for the predictions.

        Returns:
            The K-CAM loss.
        """

        if len(predictions) != len(gt_labels):
            raise ValueError("Lengths of predictions and gts must be the same.")
        if scores is not None and len(scores) != len(predictions):
            raise ValueError("Length of scores must match predictions and gts when scores is not None.")
        weights = torch.stack([torch.linalg.norm(gt, ord="fro") for gt in gt_labels], dim=0).sum(dim=0).sum(dim=0)
        if scores is None:
            squared_error = torch.stack(
                [torch.linalg.norm(prediction - gt, ord="fro") for prediction, gt in zip(predictions, gt_labels)],
                dim=0).sum(dim=0).sum(dim=0)
        else:
            squared_error = torch.stack(
                [torch.linalg.norm((prediction - gt) * sc[:, :, None], ord="fro") for prediction, gt, sc in
                 zip(predictions, gt_labels, scores)],
                dim=0).sum(dim=0).sum(dim=0)
        if return_mean:
            mean_squared_error = (squared_error / weights).mean()
            return mean_squared_error
        else:
            return squared_error / weights

    # Define a custom function to adjust the learning rate
    def lr_lambda(self, epoch):
        if epoch % 10 == 0 and epoch > 0:
            return 0.998 * self.optimizer.param_groups[0]['lr']  # 0.2% reduction
        else:
            return 1.0

    def setup_optimizer(self, transpose=None, lr=0.005, step_size=10, gamma=0.998):

        """
        Sets up the optimizer.

        Args:
            lr: The learning rate.

        Returns:
            The transpose, optimizer, and loss function.

        """
        if transpose is None:
            transpose = torch.tensor([1.1, 1.1, 1.1, 2.1], dtype=torch.float64).to(self.device)
            transpose.requires_grad = True

        optimizer = torch.optim.Adam([transpose], lr=lr)
        # Create a partial function to pass optimizer to lr_lambda
        # partial_lr_lambda = lambda epoch: self.lr_lambda(epoch, optimizer)

        # Create a learning rate scheduler using LambdaLR
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        loss_fn = self.k_cam_loss
        return transpose, optimizer, loss_fn, scheduler

    def center_projections(self, gt_projA, gt_projB):

        """
        Centers the projections.

        Args:
            gt_projA: The first set of projections.
            gt_projB: The second set of projections.

        Returns:
            The centered projections.
        """

        gt_projA = gt_projA[:, :2]
        gt_projB = gt_projB[:, :2]
        gt_cent = (gt_projA[0] + gt_projB[0]) / 2
        gt_projA = gt_projA - gt_cent
        gt_projB = gt_projB - gt_cent
        gt_x = np.concatenate([gt_projA[:, 0], gt_projB[:, 0]])
        gt_y = np.concatenate([gt_projA[:, 1], gt_projB[:, 1]])
        scale_gt = np.sqrt(gt_x ** 2 + gt_y ** 2).mean(axis=0)
        return gt_projA / scale_gt, gt_projB / scale_gt

    def normalize_projections(self, proj1, proj2, number_of_landmarks):

        """
        Normalizes the projections.

        Args:
            proj1: The first set of projections.
            proj2: The second set of projections.
            number_of_landmarks: The number of landmarks.

        Returns:
            The normalized projections.
        """

        proj1_n = [
            self.pose_data._normalize_2d(d[:, :2].reshape(-1, number_of_landmarks * 2)).astype(np.float32).reshape(
                number_of_landmarks, 2) for d in proj1]
        proj2_n = [
            self.pose_data._normalize_2d(d[:, :2].reshape(-1, number_of_landmarks * 2)).astype(np.float32).reshape(
                number_of_landmarks, 2) for d in proj2]
        proj1_n = torch.from_numpy(np.asarray(proj1_n))
        proj2_n = torch.from_numpy(np.asarray(proj2_n))
        proj_b = torch.stack([proj1_n, proj2_n], dim=1).to(self.device)
        return proj_b

    def normalize_projection(self, proj1, number_of_landmarks):

        """
        Normalizes the projections.

        Args:
            proj1: The first set of projections.
            number_of_landmarks: The number of landmarks.

        Returns:
            The normalized projections.
        """

        proj1_n = [
            self.pose_data._normalize_2d(d[:, :2].reshape(-1, number_of_landmarks * 2)).astype(np.float32).reshape(
                number_of_landmarks, 2) for d in proj1]
        proj1_n = torch.from_numpy(np.asarray(proj1_n))
        proj_b = torch.stack([proj1_n], dim=1).to(self.device)
        return proj_b

    def normalize_projection_batch(self, projections, number_of_landmarks):

        """
        Normalizes the projections in a batch.

        Args:
            projections: The projections.
            number_of_landmarks: The number of landmarks.

        Returns:
            The normalized projections and the original projections.
        """

        tmp = []
        for proj1 in projections:
            proj1_n = [
                self.pose_data._normalize_2d(d[:, :2].reshape(-1, number_of_landmarks * 2)).astype(np.float32).reshape(
                    number_of_landmarks, 2) for d in proj1]
            proj1_n = torch.from_numpy(np.asarray(proj1_n))
            tmp.append(proj1_n)
        proj_b = torch.stack(tmp, dim=1).to(self.device)
        return tmp, proj_b

    def process_cameras(self, cameras):
        """
        Processes the cameras.

        Args:
            cameras: The cameras.

        Returns:
            The processed cameras.
        """

        cameras_np = [c.detach().cpu().numpy() for c in cameras]
        nmAR = [np.zeros(3)]
        nmBR = [np.zeros(3)]

        for i in range(self.camera_count):
            nmAR.append(cameras_np[i][0, :, 0])
            nmAR.append(cameras_np[i][0, :, 1])
            nmBR.append(cameras_np[i][1, :, 0])
            nmBR.append(cameras_np[i][1, :, 1])

        nmAR = np.array(nmAR)
        nmBR = np.array(nmBR)

        rot = self.calibrate_by_procrustes(nmBR, None, nmAR)
        rot = torch.from_numpy(rot).to(self.device)
        return rot

    def run_optimization(self, pred_shapeA, pred_shapeB_r, optimizer, transpose, max_iterations, camera_count, cameras,
                         loss_fn, gt, scheduler, scores_list):

        """
        Runs the optimization.

        Args:
            pred_shapeA: The predicted shape A.
            pred_shapeB_r: The predicted shape B rotated.
            optimizer: The optimizer.
            transpose: The transpose.
            max_iterations: The maximum number of iterations.
            camera_count: The number of cameras.
            cameras: The cameras.
            loss_fn: The loss function.
            gt: The ground truth.

        Returns:
            The loss, the optimizer, and the predicted shape B.
        """

        for it in range(max_iterations):
            optimizer.zero_grad()
            pred_shapeB_t = (pred_shapeB_r * transpose[-1] + transpose[:-1])
            pred = []
            for cam_idx in range(camera_count):
                cam = cameras[cam_idx][0].float()
                ptA = (pred_shapeA.float() @ cam)
                ptB = (pred_shapeB_t.float() @ cam)
                pred_cent = (ptA[0] + ptB[0]) / 2
                ptA = ptA - pred_cent
                ptB = ptB - pred_cent
                scale_pred = torch.sqrt(
                    torch.cat([ptA[:, 0], ptB[:, 0]]) ** 2 + torch.cat([ptA[:, 1], ptB[:, 1]]) ** 2).mean(axis=0)
                ptA = ptA / scale_pred
                ptB = ptB / scale_pred
                pred.append(torch.cat([ptA, ptB]))
            loss = loss_fn(pred, gt, True)
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            # if it % 10 == 0:
            #     print(f"Iteration {it}: {scheduler.get_last_lr()}, {loss.item()}")

        return loss, optimizer, pred_shapeB_t

    def min_two_indices(self, arr):
        """
        Find the indices of the two minimum values in the array.

        Parameters:
        - arr: numpy array.

        Returns:
        - indices: tuple of two indices corresponding to the two minimum values.
        """
        sorted_indices = np.argsort(arr)
        min_indices = sorted_indices[:1]
        return list(min_indices)

    def minimize_reprojection_loss(self, pred3d, proj_gt, losses_, cameras, maxiter=100):
        """
        Minimizes the reprojection loss between predicted 3D points and ground truth projections.

        Args:
            pred3d: The predicted 3D points.
            proj_gt: The ground truth projections.
            losses_: The current loss values.

        Returns:
            The updated loss values.
        """
        cameras_np = [c.detach().cpu().numpy() for c in cameras]

        for proj_index, pred_shapeA in enumerate(pred3d):
            for cam_index in range(self.camera_count):
                points_3d = pred_shapeA.detach().cpu().numpy()
                points_2d_gt = proj_gt[proj_index][cam_index].detach().cpu().numpy()

                params = cameras_np[cam_index][proj_index]
                homogeneous_array = np.hstack((params, np.ones((params.shape[0], 1))))
                parameters = homogeneous_array.flatten()

                initial_parameters = parameters

                result = minimize(reprojection_loss, initial_parameters, args=(points_3d, points_2d_gt),
                                  method='L-BFGS-B', options={'maxiter': maxiter})

                estimated_parameters = result.x

                final_loss = reprojection_loss(estimated_parameters, points_3d
                                               , points_2d_gt)
                losses_[proj_index] += final_loss * final_loss

        return losses_

    def get_bounding_box(self, keypoints):
        """
        Retrieves the bounding box boundaries for 2D landmarks.

        Args:
            keypoints: A list of 2D landmarks.

        Returns:
            Tuple (x_min, y_min, x_max, y_max) representing the bounding box boundaries.
        """
        x_values = [keypoint[0] for keypoint in keypoints]
        y_values = [keypoint[1] for keypoint in keypoints]
        x_min = min(x_values)
        y_min = min(y_values)
        x_max = max(x_values)
        y_max = max(y_values)
        return x_min, y_min, x_max, y_max

    def generate_combinations(self, people, cameras_predictions, frame_index):

        """
        Generates combinations of people and camera predictions.

        Args:
            people: A list of people.
            cameras_predictions: A list of camera predictions.
            frame_index: The index of the frame.

        Returns:
            projections_list: A list of projections.
            bounding_box_list: A list of bounding boxes.
            assignments: A list of assignments.
        """

        projections_list = []
        bounding_box_list = []
        scores_list = []
        assignments = list(product(people, repeat=len(cameras_predictions)))

        for i, assignment in enumerate(assignments):
            projections = []
            bbox = []
            scores = []

            for camera, person in zip(cameras_predictions, assignment):
                projections.append(camera[frame_index]['instances'][person]['keypoints'])
                scores.append(camera[frame_index]['instances'][person]['keypoint_scores'])
                # bbox.append(self.get_bounding_box(camera[frame_index]['instances'][person]['keypoints']))
                bbox.append(camera[frame_index]['instances'][person]['bbox'])

            projections_list.append(np.array(projections))
            bounding_box_list.append(bbox)
            scores_list.append(scores)

        return projections_list, bounding_box_list, scores_list, assignments

    def run_pipeline(self, camera_prefix="sync_", camera_count=3, max_iterations=570, min_iterations=50,
                     iterations_step=20, save_iteration=500, save_mode=False, draw_bbox=True):

        """
           Runs the pipeline to process frames from multiple cameras.

           Args:
               camera_prefix: The prefix for camera filenames.
               camera_count: The number of cameras.
               max_iterations: The maximum number of iterations for optimization.
               min_iterations: The minimum number of iterations for optimization.
               iterations_step: The step size for changing the number of iterations.
               save_iteration: The iteration interval for saving the model.
               save_mode: Whether to save the model.
               draw_bbox: Whether to draw bounding boxes.

           Yields:
               frame_index: The index of the frame.
               pred_shapeA: The predicted shape A.
               pred_shapeB_t: The predicted shape B transformed.

           Raises:
               Exception: If an error occurs during processing.
           """
        data_path = self.subjects[0]
        final_losses = []
        final_time = []
        if self.input_type == "video":
            caps = [cv2.VideoCapture(os.path.join(data_path, f'{camera_prefix}{str(i + 1)}.mp4')) for i in
                    range(camera_count)]

            cameras_predictions = []
            for i in range(camera_count):
                cameras_predictions.append(load_json(os.path.join(data_path, f'{camera_prefix}{str(i + 1)}.json')))

        frame_index = 0

        frames = [None] * len(caps)
        rets = [None] * len(caps)
        colors = [(255, 0, 0), (0, 0, 255)]

        stat_indx = [0, 1]
        os.makedirs(self.fine_tuned_path, exist_ok=True)
        previous_a = None
        previous_b = None
        transpose = None
        while all(cap.isOpened() for cap in caps):

            try:

                for i, cap in enumerate(caps):
                    rets[i], frames[i] = cap.read()

                if not all(rets):
                    break

                lr = max(1 - (20 / max_iterations), 0.1)
                step_size = max((max_iterations // iterations_step), 5)
                gamma = max(1. - (1 / step_size) * 5, 0.6)

                transpose, optimizer, loss_fn, scheduler = self.setup_optimizer(transpose, lr=lr, step_size=step_size,
                                                                                gamma=gamma)

                # Generate all possible combinations of people assigned to cameras
                people = list(range(2))

                # Generate all possible combinations of people assigned to cameras
                projections_list, bounding_box_list, scores_list, assignments = self.generate_combinations(people,
                                                                                                           cameras_predictions,
                                                                                                           frame_index)

                scores_list = np.array(scores_list)


                proj_gt, proj_b = self.normalize_projection_batch(projections_list, self.number_of_landmarks)
                pred3d, cameras = self.trainer.net(proj_b)
                pred3d = pred3d.float()
                losses_ = [0] * len(assignments)

                losses_ = self.minimize_reprojection_loss(pred3d, proj_gt, losses_, cameras)
                # print("losses:  ",losses_)

                min_indices = self.min_two_indices(np.array(losses_))
                min_indices.append(abs(7 - min_indices[0]))
                bbox_array = np.array(bounding_box_list)[sorted(min_indices)]
                scores_list = scores_list[sorted(min_indices)]

                if len(bbox_array.shape) > 3:
                    reshaped_bbox_array = bbox_array.transpose(1, 0, 2, 3).reshape(
                        bbox_array.shape[1], -1, 4)
                else:
                    reshaped_bbox_array = bbox_array.transpose(1, 0, 2)

                selected_people = np.array(projections_list)[sorted(min_indices)]
                if previous_a is not None:
                    sim1 = compute_similarity(previous_a, selected_people[0])
                    sim2 = compute_similarity(previous_b, selected_people[1])

                    sima = compute_similarity(previous_a, selected_people[1])
                    simb = compute_similarity(previous_b, selected_people[0])

                    if sim1 + sim2 < sima + simb:
                        colors[0], colors[1] = colors[1], colors[0]
                        stat_indx[0], stat_indx[1] = stat_indx[1], stat_indx[0]
                        print("Swapped: ", frame_index)
                        print(colors)

                previous_a, previous_b = selected_people[0], selected_people[1]
                if draw_bbox:
                    # Create a thread for processing the bounding boxes
                    thread = threading.Thread(target=draw_bounding_boxes, args=(
                        frames, reshaped_bbox_array, self.output_visualization, colors,
                        frame_index))

                    # Start the thread
                    thread.start()

                projections1 = selected_people[stat_indx[0]]
                projections2 = selected_people[stat_indx[1]]

                projections1 = np.array(projections1)
                projections2 = np.array(projections2)

                gt_projsA = []
                gt_projsB = []

                for gt_projA, gt_projB in zip(projections1, projections2):
                    a, b = self.center_projections(gt_projA, gt_projB)
                    gt_projsA.append(a)
                    gt_projsB.append(b)

                gt_projsA = torch.from_numpy(np.asarray(gt_projsA)).to(self.device)
                gt_projsB = torch.from_numpy(np.asarray(gt_projsB)).to(self.device)

                proj_b = self.normalize_projections(projections1, projections2, self.number_of_landmarks)
                pred3d, cameras = self.trainer.net(proj_b)
                pred3d = pred3d.float()

                rot = self.process_cameras(cameras)

                pred_shapeA = pred3d[0]
                pred_shapeB_r = (pred3d[1].float() @ rot.float())
                gt = [torch.cat([a, b]) for a, b in zip(gt_projsA, gt_projsB)]

                start_timer = time.time()
                loss, optimizer, pred_shapeB_t = self.run_optimization(pred_shapeA, pred_shapeB_r, optimizer, transpose,
                                                                       max_iterations,
                                                                       camera_count, cameras,
                                                                       loss_fn, gt, scheduler, scores_list)

                print('loss', loss.detach().cpu().numpy().item())
                # get the value of torch tensor as float

                final_losses.append(loss.detach().cpu().numpy().item())

                end_timer = time.time()
                final_time.append(end_timer - start_timer)
                print(f"Total {frame_index} frame time", end_timer - start_timer)
                yield frame_index, pred_shapeA, pred_shapeB_t

                if max_iterations > min_iterations:
                    max_iterations -= iterations_step
                else:
                    max_iterations = min_iterations

                if frame_index % save_iteration == 0:
                    if save_mode:
                        torch.save(self.trainer.net.state_dict(), f"{self.fine_tuned_path}/{str(frame_index)}.pth")
            except:
                print("Exception for image index: ", frame_index)

            finally:
                frame_index += 1
                # print(f"Total {frame_index} frame time", end_timer - start_timer)
                print(f"number of iterations: {max_iterations} ")

        if save_mode:
            torch.save(self.trainer.net.state_dict(), f"{self.fine_tuned_path}/final_model.pth")
        print("Final losses: ", final_losses)
        print("Final time: ", final_time)


# Example usage of the PoseEstimationPipeline class
if __name__ == "__main__":

    save_images = True
    pipeline = PoseEstimationPipeline()
    pipeline.load_network()
    for idx, pred_shapeA, pred_shapeB_t in pipeline.run_pipeline(camera_count = pipeline.camera_count):
        if save_images:
            thread = threading.Thread(target=export_frames_as_images, args=(
                [pred_shapeA.detach().cpu().numpy(), pred_shapeB_t.detach().cpu().numpy()], None,
                f"output/3d/{pipeline.experiment_id}", 2, [pipeline.landmarks_connections], idx))
            thread.start()
