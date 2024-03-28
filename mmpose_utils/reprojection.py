import numpy as np
from scipy.optimize import minimize


def project_3d_to_2d(points_3d, parameters):
    """
    Projects 3D points to 2D using the camera parameters.

    Args:
        points_3d: numpy array of shape (N, 3), where N is the number of 3D points.
        parameters: numpy array of shape (6,), representing [rotation, translation].

    Returns:
        numpy array of shape (N, 2): The projected 2D points.
    """

    rotation_matrix = np.reshape(parameters[:9], (3, 3))
    translation_vector = np.zeros(3) #parameters[9:] # #

    # Create a 3x4 transformation matrix [R | t]
    # print(rotation_matrix)
    # print(translation_vector)
    transformation_matrix = np.column_stack((rotation_matrix, translation_vector)) #rotation_matrix

    # Homogeneous coordinates of 3D points
    homogeneous_coordinates =np.column_stack((points_3d, np.ones((points_3d.shape[0], 1)))) # points_3d

    # Project 3D points to 2D
    points_2d_homogeneous = np.dot(transformation_matrix, homogeneous_coordinates.T).T
    points_2d = points_2d_homogeneous[:, :2]

    return points_2d


def reprojection_loss(parameters, points_3d, points_2d_gt):
    """
    Computes reprojection loss between projected 3D points and ground truth 2D points.

    Args:
        parameters: numpy array of shape (12,), representing [rotation, translation].
        points_3d: numpy array of shape (N, 3), where N is the number of 3D points.
        points_2d_gt: numpy array of shape (N, 2), ground truth 2D points.

    Returns:
        Scalar: The reprojection loss.
    """
    points_2d_projected = project_3d_to_2d(points_3d, parameters)
    loss = np.mean(np.linalg.norm(points_2d_projected - points_2d_gt, axis=1))
    return loss


if __name__ == '__main__':
    # Example usage:
    # Define your 3D points and ground truth 2D points
    points_3d = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    points_2d_gt = np.array([[100, 100], [201, 200], [300, 300]])
    # Initial guess for parameters [rotation, translation]
    initial_parameters = np.zeros(9)
    # Minimize reprojection loss to estimate parameters
    # change the minimize method to do only 10 iterations
    result = minimize(reprojection_loss, initial_parameters, args=(points_3d, points_2d_gt), method='L-BFGS-B', options={'maxiter': 15})
    # result = minimize(reprojection_loss, initial_parameters, args=(points_3d, points_2d_gt), method='L-BFGS-B')
    # # Extract the estimated parameters
    estimated_parameters = result.x
    print(estimated_parameters)
    # Compute reprojection loss using the estimated parameters
    final_loss = reprojection_loss(estimated_parameters, points_3d, points_2d_gt)
    print("Estimated Parameters:", estimated_parameters)
    print("Final Reprojection Loss:", final_loss)
