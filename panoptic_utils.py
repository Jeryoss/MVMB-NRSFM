import os
import sys

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import torch

sys.path.append('.')
sys.path.append('..')

import utils.panoptic as panutils
from utils.motion_capture import get_trace3d
from utils.reconstruction import calibrate_by_procrustes
import pickle
import logging
import numpy as np

from tqdm import tqdm
from torch.optim import Adam, lr_scheduler

BLUE = "rgb(90, 130, 238)"
RED = "rgb(205, 90, 76)"
GREEN = "rgb(10, 200, 76)"
CONNECTIONS_PANOPTIC = np.array(
    [[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9], [3, 13], [13, 14], [14, 15], [1, 10], [10, 11],
     [11, 12]]) - 1
colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
body_edges = np.array(
    [[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9], [3, 13], [13, 14], [14, 15], [1, 10], [10, 11],
     [11, 12]]) - 1
arrow_edges = np.array([[0, 19], [0, 20], [0, 21]])


def calculate_jointwise_distance(predictions_array, ground_truth_array):
    """
    Calculate the average Euclidean distance between corresponding points in two sets of 3D points.

    Args:
        predictions_array (numpy.ndarray): The first set of 3D points (predicted positions).
        ground_truth_array (numpy.ndarray): The second set of 3D points (ground truth positions).

    Returns:
        float: The average Euclidean distance between corresponding points in the predicted and ground truth sets.
    """
    try:
        error_array = np.zeros(len(predictions_array))
        for idx, (gt, pred) in enumerate(zip(ground_truth_array, predictions_array)):
            error_array[idx] = np.linalg.norm(pred - gt)
        error = np.mean(error_array)
        logging.debug("Successfully calculated jointwise distance.")
        return error
    except Exception as e:
        logging.error(f"Failed to calculate jointwise distance: {e}")
        raise


def calculate_error_rate(data):
    """
    Calculate the average jointwise distance error for all poses in the data.

    This function iterates over all the poses in the data, calculates the jointwise distance error for each pose, and then takes the average of these errors.

    Args:
        data (dict): The data dictionary containing all necessary information.

    """
    try:
        error_list = []
        for index in range(0, len(data["shapes_procrustes"])):
            pred = data["shapes_procrustes"][index] * data["scales"][index][0]
            gt = data["groundtruth"][index][0] * data["scales"][index][0]

            error = calculate_jointwise_distance(pred, gt)
            error_list.append(error)
        error = np.mean(np.asarray(error_list))

        print('--------single pose---------', error)
        logging.info("Successfully calculated error rate.")
    except Exception as e:
        logging.error(f"Failed to calculate error rate: {e}")
        raise


def get_connected_samples(data):
    """
    Get connected samples in the data.

    This function iterates over a subset of the data and for each item, it finds all other items that have the same sequence name and frame index. It then appends these connected samples to a list. If a set of connected samples is not already in the list, it is added.

    Args:
        data (dict): The data dictionary containing all necessary information.

    Returns:
        list: A list of connected samples.
    """
    try:
        connected_samples_Frames = []
        for index in range(len(data["frame_ideces"])):

            connected_samples = []
            for i, (s, f) in enumerate(zip(data["seq_names"], data["frame_ideces"])):
                if data["seq_names"][index] == s and data["frame_ideces"][index] == f:
                    connected_samples.append(i)
            if connected_samples not in connected_samples_Frames:
                connected_samples_Frames.append(connected_samples)
        logging.debug("Successfully retrieved connected samples.")
        return connected_samples_Frames
    except Exception as e:
        logging.error(f"Failed to get connected samples: {e}")
        raise


def rotation_matrix(axis, theta):
    """
    Returns the rotation matrix associated with a counterclockwise rotation about a given axis by theta radians.

    This function generates a 3D rotation matrix that represents a rotation about a given axis by a specified angle.
    This rotation matrix can be used to rotate a point or vector in 3D space.

    Args:
        axis (numpy.ndarray): A 3D vector that represents the axis of rotation.
        theta (float): The angle of rotation in radians.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.

    """
    try:
        axis = np.asarray(axis)
        axis /= np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
        # logging.info("Successfully calculated rotation matrix.")
        return rotation_matrix
    except Exception as e:
        logging.error(f"Failed to calculate rotation matrix: {e}")
        raise


def load_image(data_path, seq_name, cam, hd_idx):
    """
    Load an image from the specified path.

    Args:
        data_path (str): The path to the data.
        seq_name (str): The sequence name.
        cam (dict): The camera details.
        hd_idx (int): The HD index.

    Returns:
        ndarray: The loaded image.
    """
    hd_img_path = os.path.join(data_path, seq_name, 'hdImgs',
                               '{0:02d}_{1:02d}/{0:02d}_{1:02d}_{2:08d}.jpg'.format(cam['panel'], cam['node'], hd_idx))
    return plt.imread(hd_img_path)


def add_camera_view(data, data_path, fig, index, connected_samples, cam_idx):
    """
    Add a camera view to the given figure.

    Args:
        data (dict): The data dictionary containing all necessary information.
        data_path (str): The path to the data.
        fig (matplotlib.figure.Figure): The figure to add the subplot to.
        index (int): The index of the current data point.
        connected_samples (list): The list of connected samples.
        cam_idx (int): The index of the camera.

    Raises:
        Exception: If there is an error in adding the camera view.
    """
    try:
        fig.add_subplot(3, 1, cam_idx + 1)
        cam = data["gt_cameras"][index][cam_idx]
        seq_name = data["seq_names"][index]
        hd_idx = data["frame_ideces"][index]
        im = load_image(data_path, seq_name, cam, hd_idx)

        plt.title('3D Body Projection on HD view ({0})'.format(cam['name']))
        plt.imshow(im)
        currentAxis = plt.gca()
        currentAxis.set_autoscale_on(False)

        for i, si in enumerate(connected_samples):
            skel = data["poses"][si].transpose()
            pt = panutils.projectPoints(skel,
                                        cam['K'], cam['R'], cam['t'],
                                        cam['distCoef'])

            plt.plot(pt[0, :], pt[1, :], '.', color=colors[i])

            # Plot edges for each bone
            for edge in body_edges:
                plt.plot(pt[0, edge], pt[1, edge], color=colors[i])
        logging.info("Successfully added camera view.")
    except Exception as e:
        logging.error(f"Failed to add camera view: {e}")
        raise


def get_figure3d(points3ds, gts=None, range_scale=1, connections=CONNECTIONS_PANOPTIC):
    """Yields plotly fig for visualization"""
    traces = []
    for points3d in points3ds:
        traces += get_trace3d(points3d, BLUE, BLUE, "prediction", connections=connections)
    if gts is not None:
        for gt in gts:
            traces += get_trace3d(gt, RED, RED, "groundtruth", connections=connections)
    layout = go.Layout(
        scene=dict(
            aspectratio=dict(x=0.8,
                             y=0.8,
                             z=2),
            xaxis=dict(range=(-0.5 * range_scale, 0.5 * range_scale), ),
            yaxis=dict(range=(-0.5 * range_scale, 0.5 * range_scale), ),
            zaxis=dict(range=(-1 * range_scale, 1 * range_scale), ), ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))
    return go.Figure(data=traces, layout=layout)


def calibrate_by_procrustes(points3d, camera, gt, normalize=False):
    """
    Calibrates the predicted 3D points by Procrustes algorithm.

    This function estimates an orthonormal matrix for aligning the predicted 3D
    points to the ground truth. This orthonormal matrix is computed by
    Procrustes algorithm, which ensures the global optimality of the solution.

    Args:
        points3d (numpy.ndarray): The predicted 3D points.
        camera (numpy.ndarray): The camera details.
        gt (numpy.ndarray): The ground truth 3D points.
        normalize (bool): Whether to normalize the points3d and gt before applying the Procrustes algorithm.

    Returns:
        numpy.ndarray: The orthonormal matrix for aligning the predicted 3D points to the ground truth.
    """
    try:
        # Shift the center of points3d to the origin
        if normalize:
            if camera is not None:
                singular_value = np.linalg.norm(camera, 2)
                camera = camera / singular_value
                points3d = points3d * singular_value
            scale = np.linalg.norm(gt) / np.linalg.norm(points3d)
            points3d = points3d * scale

        U, s, Vh = np.linalg.svd(points3d.T.dot(gt))
        rot = U.dot(Vh)
        return rot
    except Exception as e:
        logging.error(f"Failed to calibrate by procrustes: {e}")
        raise


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2.

    Args:
        vec1 (numpy.ndarray): A 3d "source" vector.
        vec2 (numpy.ndarray): A 3d "destination" vector.

    Returns:
        numpy.ndarray: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    try:
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        logging.debug("Successfully calculated rotation matrix.")
        return rotation_matrix
    except Exception as e:
        logging.error(f"Failed to calculate rotation matrix: {e}")
        raise


def normalize_3d(pose):
    """
    Normalize a 3D pose.

    This function takes a 3D pose as input and normalizes it. It subtracts the x, y, and z coordinates of the first point in the pose from all points in the pose. Then, it calculates the mean length of the vectors from the first point to all other points in the pose and divides all points in the pose by this mean length.

    Args:
        pose (numpy.ndarray): The 3D pose to normalize.

    Returns:
        tuple: The normalized pose and the mean length.
    """
    try:
        xs = pose.T[0::3] - pose.T[0]
        ys = pose.T[1::3] - pose.T[1]
        ls = np.sqrt(xs[1:19] ** 2 + ys[1:19] ** 2)
        scale = ls.mean(axis=0)
        pose = pose.T / scale
        pose[0::3] -= pose[0].copy()
        pose[1::3] -= pose[1].copy()
        pose[2::3] -= pose[2].copy()
        return pose.T, scale
    except Exception as e:
        logging.error(f"Failed to normalize 3D pose: {e}")
        raise


def normalize_3d_2P(pose1, pose2):
    """
    Normalize two 3D poses.

    This function takes two 3D poses as input and normalizes them. It first calculates the center point of the two poses and subtracts this center point from all points in both poses. Then, it calculates the mean length of the vectors from the first point to all other points in both poses and divides all points in both poses by this mean length.

    Args:
        pose1 (numpy.ndarray): The first 3D pose to normalize.
        pose2 (numpy.ndarray): The second 3D pose to normalize.

    Returns:
        tuple: The two normalized poses and the mean length.
    """
    try:
        pose_cent = (pose1[0] + pose2[0]) / 2

        pose1 = pose1 - pose_cent
        pose2 = pose2 - pose_cent

        poses_x = np.concatenate([pose1[:, 0], pose2[:, 0]])

        poses_y = np.concatenate([pose1[:, 1], pose2[:, 1]])

        poses_z = np.concatenate([pose1[:, 2], pose2[:, 2]])

        scale_p = np.sqrt(poses_x ** 2 + poses_y ** 2 + poses_z ** 2).mean(axis=0)
        pose1 = pose1 / scale_p
        pose2 = pose2 / scale_p
        return pose1, pose2, scale_p
    except Exception as e:
        logging.error(f"Failed to normalize 3D poses: {e}")
        raise


def k_cam_loss(predictions, gt_labels, return_mean=True, scores=None):
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


def optimize_translation(cfg, n, pred_shapeA_tr, pred_shapeB_tr, gt_projsA, gt_projsB, pred_camAs_t):
    """
    Optimize the translation of the second 3D pose to align it with the first 3D pose.

    This function minimizes the K-CAM loss between the projected poses and the ground truth projections by using Adam as the optimization algorithm. It initializes a 3D translation vector and updates it in each iteration to minimize the K-CAM loss. The updated translation vector is then added to the second 3D pose to align it with the first 3D pose.

    Args:
        cfg: The configuration object that contains various settings.
        n: The number of iterations for the optimization process.
        pred_shapeA_tr: The first 3D pose.
        pred_shapeB_tr: The second 3D pose.
        gt_projsA: The ground truth projections for the first pose.
        gt_projsB: The ground truth projections for the second pose.
        pred_camAs_t: The predicted camera matrices for the first pose.

    Returns:
        The translated second 3D pose.
    """
    try:

        # transpose = torch.tensor([0.1, 0.1, 0.1, 1.], dtype=torch.float64)
        transpose = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        transpose.requires_grad = True

        # Initialize Adam optimizer with weight decay
        optimizer = Adam([transpose], lr=0.1, weight_decay=0.01)

        # Initialize learning rate scheduler to reduce learning rate by 5% each epoch
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

        loss_fn = k_cam_loss

        n_iter = n  # fix the number of iterations
        for it in range(n_iter):
            optimizer.zero_grad()

            pred_shapeB_tr_t = pred_shapeB_tr + transpose
            pred = []
            gt = [torch.cat([a, b]) for a, b in zip(gt_projsA, gt_projsB)]

            for cam_idx in range(cfg.n_cams):
                ptA = (pred_shapeA_tr @ pred_camAs_t[cam_idx])
                ptB = (pred_shapeB_tr_t @ pred_camAs_t[cam_idx])
                pred_cent = (ptA[0] + ptB[0]) / 2
                ptA = ptA - pred_cent
                ptB = ptB - pred_cent

                scale_pred = torch.sqrt(
                    torch.cat([ptA[:, 0], ptB[:, 0]]) ** 2 + torch.cat([ptA[:, 1], ptB[:, 1]]) ** 2).mean(axis=0)
                ptA = ptA / scale_pred
                ptB = ptB / scale_pred

                pred.append(torch.cat([ptA, ptB]))

            loss = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()

            # Step the learning rate scheduler
            scheduler.step()

        logging.info("Successfully optimized translation.")
        return pred_shapeB_tr_t
    except Exception as e:
        logging.error(f"Failed to optimize translation: {e}")
        raise


def calculate_reference_pose(reference_body, other_body, data, cfg):
    pred_camAs = data["cameras"][reference_body]
    pred_camBs = data["cameras"][other_body]

    nmAR = []
    nmBR = []
    n_cams = cfg.n_cams

    for cam_idx in range(n_cams):
        nmAR.append(pred_camAs[cam_idx][:, 0])
        nmAR.append(pred_camAs[cam_idx][:, 1])
        nmBR.append(pred_camBs[cam_idx][:, 0])
        nmBR.append(pred_camBs[cam_idx][:, 1])

    nmAR = np.array(nmAR)
    nmBR = np.array(nmBR)

    rot = calibrate_by_procrustes(nmBR, None, nmAR)

    pred_shapeA = data["shapes"][reference_body]
    pred_shapeB = data["shapes"][other_body]

    pred_shapeB_r = pred_shapeB @ rot

    gt_projsA = []
    gt_projsB = []

    for cam_idx in list(range(cfg.n_cams)):
        cam = data["gt_cameras"][reference_body][cam_idx]

        gt_skel = data["poses"][reference_body].transpose()
        gt_projA = panutils.projectPoints(gt_skel,
                                          cam['K'], cam['R'], cam['t'],
                                          cam['distCoef'])[0:2, :].T

        gt_skelB = data["poses"][other_body].transpose()
        gt_projB = panutils.projectPoints(gt_skelB,
                                          cam['K'], cam['R'], cam['t'],
                                          cam['distCoef'])[0:2, :].T

        gt_cent = (gt_projA[0] + gt_projB[0]) / 2
        gt_projA = gt_projA - gt_cent
        gt_projB = gt_projB - gt_cent

        gt_x = np.concatenate([gt_projA[:, 0], gt_projB[:, 0]])
        gt_y = np.concatenate([gt_projA[:, 1], gt_projB[:, 1]])
        scale_gt = np.sqrt(gt_x ** 2 + gt_y ** 2).mean(axis=0)

        gt_projsA.append(gt_projA / scale_gt)
        gt_projsB.append(gt_projB / scale_gt)



    gt_projsA = torch.from_numpy(np.asarray(gt_projsA))
    gt_projsB = torch.from_numpy(np.asarray(gt_projsB))

    pred_camAs_t = torch.from_numpy(pred_camAs).type(torch.float64)

    pred_shapeA_tr = torch.from_numpy(pred_shapeA).type(torch.float64)
    pred_shapeB_tr = torch.from_numpy(pred_shapeB_r).type(torch.float64)
    pred_shapeB_tr_t = optimize_translation(cfg, 200, pred_shapeA_tr, pred_shapeB_tr, gt_projsA, gt_projsB,
                                            pred_camAs_t)

    cam = data["gt_cameras"][reference_body][0]
    gt_poseA = data["poses"][reference_body]
    gt_poseB = data["poses"][other_body]

    pred_shapeA_tr = pred_shapeA_tr.detach().numpy()
    pred_shapeB_tr_t = pred_shapeB_tr_t.detach().numpy()

    # X_A for 2
    X_a = cam['R'].dot(gt_poseA.T).T
    X_b = cam['R'].dot(gt_poseB.T).T
    X_a = np.asarray(X_a)
    X_b = np.asarray(X_b)
    X_a, X_b, the_scale = normalize_3d_2P(X_a, X_b)
    pred_shapeA_tr, pred_shapeB_tr_t, _ = normalize_3d_2P(pred_shapeA_tr, pred_shapeB_tr_t)
    # concatenate with 2 Person
    X_ab = np.concatenate([X_a, X_b], 0)
    pred_shapeAB_tr = np.concatenate([pred_shapeA_tr, pred_shapeB_tr_t], 0)
    # procrustes with 2 Person
    pro_rot_AB = calibrate_by_procrustes(pred_shapeAB_tr, None, X_ab)

    pred_A_pr_AB = pred_shapeA_tr @ pro_rot_AB
    pred_B_pr_AB = pred_shapeB_tr_t @ pro_rot_AB

    pred_A_pr_AB = pred_A_pr_AB * the_scale
    pred_B_pr_AB = pred_B_pr_AB * the_scale

    X_a = X_a * the_scale
    X_b = X_b * the_scale

    return pred_A_pr_AB, pred_B_pr_AB, X_a, X_b


def train_video(connected_samples_Frames, data, cfg, save_path='eva_cam', file_name='5_cam.pkl'):
    """
    Train the model on a video.

    This function takes a list of connected samples, a data dictionary, a configuration object, a save path, and a file name as arguments. For each connected sample, it calls the `CalRefPose` function to calibrate the poses and appends the result to a list. Finally, it saves this list to a file.

    Args:
        connected_samples_Frames (list): The list of connected samples.
        data (dict): The data dictionary containing all necessary information.
        cfg: The configuration object that contains various settings.
        save_path (str): The path to save the result file.
        file_name (str): The name of the result file.

    Returns:
        list: A list of dictionaries, each contains the calibration result for a pair of poses.
    """
    try:
        frame_errors = []

        for connected_samples in tqdm(connected_samples_Frames):
            print(connected_samples)
            reference_body = connected_samples[0]
            predictions = np.array([])
            ground_truths = np.array([])
            for other_body in connected_samples[1:]:
                pred_A_pr_AB, pred_B_pr_AB, X_a, X_b = calculate_reference_pose(reference_body, other_body, data,
                                                                                     cfg)
                if len(predictions) == 0:
                    predictions = np.array([pred_A_pr_AB, pred_B_pr_AB])
                else:
                    predictions = np.append(predictions, [pred_B_pr_AB], axis=0)

                if len(ground_truths) == 0:
                    ground_truths = np.array([X_a, X_b])
                else:
                    ground_truths = np.append(ground_truths, [X_b], axis=0)

            predictions = predictions.reshape(-1, predictions.shape[-1])
            ground_truths = ground_truths.reshape(-1, ground_truths.shape[-1])
            frame_error = calculate_jointwise_distance(predictions, ground_truths)
            frame_errors.append(frame_error)
        print('frame_errors', sum(frame_errors) / len(frame_errors))
        logging.info("Successfully trained the video.")

    except Exception as e:
        logging.error(f"Failed to train the video: {e}")
        raise


def calculate_video_error(ALL_inf):
    """
    Calculate the average error for the calibration of poses in a video.

    This function takes a list of dictionaries (each dictionary contains the calibration result for a pair of poses) as an argument. It extracts the average error from each dictionary and calculates the mean of these errors.

    Args:
        ALL_inf (list): A list of dictionaries, each contains the calibration result for a pair of poses.

    Returns:
        float: The average error for the calibration of poses in the video.
    """
    try:
        avg_error_AB = []

        for dic in ALL_inf:
            avg_error_AB.append(dic['avg_error_AB'])

        avg_error = np.mean(np.asarray(avg_error_AB))
        logging.info(f"Successfully calculated video error: {avg_error}")
        return avg_error
    except Exception as e:
        logging.error(f"Failed to calculate video error: {e}")
        raise
