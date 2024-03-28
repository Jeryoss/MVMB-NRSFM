import numpy as np
import pandas as pd




def calibrate_by_procrustes(points3d, camera, gt, others):
    """Calibrates the predictied 3d points by Procrustes algorithm.

    This function estimate an orthonormal matrix for aligning the predicted 3d
    points to the ground truth. This orhtonormal matrix is computed by
    Procrustes algorithm, which ensures the global optimality of the solution.
    """
    # Shift the center of points3d to the origin
    if camera is not None:
        singular_value = np.linalg.norm(camera, 2)
        camera = camera / singular_value
        points3d = points3d * singular_value
    scale = np.linalg.norm(gt) / np.linalg.norm(points3d)
    points3d = points3d * scale
    
    U, s, Vh = np.linalg.svd(points3d.T.dot(gt))
    rot = U.dot(Vh)
    if others is None:
        if camera is not None:
            return points3d.dot(rot), rot.T.dot(camera)
        else:
            return points3d.dot(rot), None
    else:
        tr_others = []
        for o in others:
            o *= scale
            o = o.dot(rot)
            tr_others.append(o)
            
        if camera is not None:
            return tr_others, rot.T.dot(camera)
        else:
            return tr_others

def evaluate_reconstruction(predictions, inputs, save=False, error_dir=None, result_dir=None):
    
    projections = []
    origprojections = []
    shapes = []
    shapes_procrustes = []
    cameras = []
    shapes_gt = []
    seq_names = []
    gt_cameras = []
    frame_ideces = []
    body_ideces = []
    scales = []
    poses = []
    us = []
    vs = []

    metrics = {
        "error_2d": [],
        "error_3d": [],
        "error_abs": []}

    for pred, inp in zip(predictions, inputs):
        
        
        points3d, camera = calibrate_by_procrustes(
            pred["pred3d"], pred["predCameras"][0], inp["points3d"][0],None)

        points2d = points3d.dot(camera)

        # Evaluates error metrics
        metrics["error_2d"].append(np.linalg.norm(
            points2d - inp["points2d"][0]) / np.linalg.norm(inp["points2d"][0]))
        metrics["error_3d"].append(np.linalg.norm(
            points3d - inp["points3d"][0]) / np.linalg.norm(inp["points3d"][0]))
        metrics["error_abs"].append(np.linalg.norm(
            points3d - inp["points3d"][0], axis=1).mean())

        # Collects predictions
        projections.append(inp["points2d"])
        origprojections.append(inp["origpoints2d"])
        
        cameras.append(pred["predCameras"])
        
        shapes_gt.append(inp["points3d"])
        shapes.append(pred["pred3d"])
        shapes_procrustes.append(points3d)
        seq_names.append(inp["seq_name"])
        frame_ideces.append(inp["frame_idx"])
        body_ideces.append(inp["body_idx"])
        gt_cameras.append(inp["gt_cameras"])
        scales.append(inp["scales"])
        poses.append(inp["pose"])

    dataframe = pd.DataFrame(metrics)


        

    if save:
        dataframe.to_csv(error_dir)

        np.savez(result_dir,
                shapes=shapes,
                cameras=cameras,
                groundtruth=shapes_gt,
                projections=projections)
    else:
        result = dict(shapes=shapes,
                        shapes_procrustes=shapes_procrustes,
                        cameras=cameras,
                        us=us,
                        vs=vs,
                        groundtruth=shapes_gt,
                        projections=projections,
                        origprojections=origprojections,
                        gt_cameras=gt_cameras,
                        seq_names=seq_names,
                        frame_ideces=frame_ideces,
                        body_ideces=body_ideces,
                        scales = scales,
                        poses= poses
                        )

        return dataframe, result