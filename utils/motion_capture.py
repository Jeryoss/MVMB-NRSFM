import os
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import math
import utils.panoptic as panutils

'''
NUM_POINTS = 31
METER_SCALER = 0.001
CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (0, 6), (6, 7), (7, 8),
               (8, 9), (8, 10), (0, 11), (11, 12), (12, 13), (13, 14), (14, 15),
               (15, 16), (13, 24), (24, 25), (25, 26), (26, 27), (27, 30),
               (27, 28), (27, 29), (13, 17), (17, 18), (18, 19), (19, 20),
               (20, 21), (20, 22), (20, 23))
'''

CONNECTIONS = [[10, 9], [9, 8], [8, 14],
               [14, 15], [15, 16], [8, 11],
               [11, 12], [12, 13], [8, 7],
               [7, 0], [1, 0], [1, 2],
               [2, 3], [0, 4], [4, 5], [5, 6]],
BLUE = "rgb(90, 130, 238)"
RED = "rgb(205, 90, 76)"
GREEN = "rgb(10, 200, 76)"
colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()

body_edges = np.array(
    [[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9], [3, 13], [13, 14], [14, 15], [1, 10], [10, 11],
     [11, 12]]) - 1
arrow_edges = np.array([[0, 19], [0, 20], [0, 21]])

CONNECTIONS_PANOPTIC = np.array(
    [[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9], [3, 13], [13, 14], [14, 15], [1, 10],
     [10, 11], [11, 12]]) - 1


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def normalize_3d(pose):
    xs = pose.T[0::3] - pose.T[0]
    ys = pose.T[1::3] - pose.T[1]
    ls = np.sqrt(xs[1:19] ** 2 + ys[1:19] ** 2)
    scale = ls.mean(axis=0)
    pose = pose.T / scale
    pose[0::3] -= pose[0].copy()
    pose[1::3] -= pose[1].copy()
    pose[2::3] -= pose[2].copy()
    return pose.T, scale

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def add_camera_view(fig, index, connected_samples, cam_idx, data_path, data):
    fig.add_subplot(3, 1, cam_idx + 1)
    cam = data["gt_cameras"][index][cam_idx]
    seq_name = data["seq_names"][index]
    hd_idx = data["frame_ideces"][index]
    hd_img_path = os.path.join(data_path, data["seq_names"][index], 'hdImgs',
                               '{0:02d}_{1:02d}/{0:02d}_{1:02d}_{2:08d}.jpg'.format(cam['panel'], cam['node'], hd_idx))
    im = plt.imread(hd_img_path)

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


def get_figure3d(points3ds, gts=None, cameras=None, range_scale=1, connections=CONNECTIONS_PANOPTIC, colors= [RED, GREEN, BLUE ]):
    """Yields plotly fig for visualization"""
    traces = []

    for c, points3d in enumerate(points3ds):
        if colors is None:
            color = BLUE
        else:
            color = colors[c]
        traces += get_trace3d(points3d, color, color,  "prediction", connections=connections)
    if gts is not None:
        for gt in gts:
            traces += get_trace3d(gt, RED, RED, "groundtruth", connections=connections)

    if cameras is not None:
        traces.append(get_camera_trace3d(cameras, GREEN))
    layout = go.Layout(
        scene=dict(
            aspectratio=dict(x=0.8,
                             y=0.8,
                             z=3.),
            xaxis=dict(range=(-0.5 * range_scale, 0.5 * range_scale), ),
            yaxis=dict(range=(-0.5 * range_scale, 0.5 * range_scale), ),
            zaxis=dict(range=(-1 * range_scale, 1 * range_scale), ), ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))
    return go.Figure(data=traces, layout=layout)


def get_camera_trace3d(camera, point_color=None, name="Camera"):
    if point_color is None:
        point_color = "rgb(30, 20, 160)"

    trace_of_points = go.Scatter3d(
        x=camera[:, 0],
        y=camera[:, 2],
        z=camera[:, 1],

        mode="markers",
        name=name,
        marker=dict(
            symbol="square",
            size=6,
            color=point_color))
    return trace_of_points


def get_trace3d(points3d, point_color=None, line_color=None, name="PointCloud", connections=CONNECTIONS):
    """Yields plotly traces for visualization.

    Args:
        points3d: A 3D point cloud with shape [N, 3].
        point_color: The color of points.
        line_color: The color of lines.
        name: The name of the trace.
        connections: The connections of the point cloud.

    Returns:
        A list of plotly traces.
    """

    if point_color is None:
        point_color = "rgb(30, 20, 160)"
    if line_color is None:
        line_color = "rgb(30, 20, 160)"
    # Trace of points.
    trace_of_points = go.Scatter3d(
        x=points3d[:, 0],
        y=points3d[:, 1],
        z=points3d[:, 2],
        mode="markers",
        name=name,
        marker=dict(
            symbol="circle",
            size=3,
            color=point_color))
    # Trace of lines.
    xlines = []
    ylines = []
    zlines = []
    for line in connections[0]:
        for point in line:
            xlines.append(points3d[point, 0])
            ylines.append(points3d[point, 1])
            zlines.append(points3d[point, 2])
        xlines.append(None)
        ylines.append(None)
        zlines.append(None)
    trace_of_lines = go.Scatter3d(
        x=xlines,
        y=ylines,
        z=zlines,
        mode="lines",
        name=name,
        line=dict(color=line_color))
    return [trace_of_points, trace_of_lines]



def normalize_points(points3ds):
    """
    Normalize the given points between -1 and 1.
    Args:
        points3ds (list of array-like): List of 3D points to normalize.
    Returns:
        list of array-like: Normalized points.
    """
    min_val = np.min(points3ds)
    max_val = np.max(points3ds)
    normalized_points = (points3ds - min_val) / (max_val - min_val) * 2 - 1
    return normalized_points

import plotly.offline as py

def export_frames_as_images(points3ds, gts=None, output_folder="output", range_scale=1, connections=CONNECTIONS,
                            index=0):
    """
    Export frames as images using Plotly and save them to the specified output folder.

    Args:
        frames (list of list of array-like): List of frames, where each frame is a list of 3D points to visualize.
        output_folder (str): The folder where images will be saved.
        range_scale (float, optional): Scaling factor for adjusting the plot range.
        connections (list of tuple, optional): List of connections between points.
    """

    points3ds = normalize_points(points3ds)
    os.makedirs(output_folder, exist_ok=True)

    figure = get_figure3d(points3ds, gts, range_scale=range_scale, connections=connections)
    image_path = os.path.join(output_folder, f"frame_{index:04d}.png")

    if index in list(range(0, 10)):
        py.iplot(figure, filename='3d-point-cloud')

    # Export the figure as an image
    figure.write_image(image_path)

    # print("Frames exported as images.")


def get_camera_trace3d(camera, point_color=None, name="Camera"):
    """
    Generate a Plotly trace for 3D visualization of a camera's position.

    Args:
        camera (numpy.ndarray): 3D camera position to visualize.
        point_color (str, optional): Color of the camera point. Default is "rgb(30, 20, 160)".
        name (str, optional): Name of the trace. Default is "Camera".

    Returns:
        plotly.graph_objs.Scatter3d: A trace object for 3D camera visualization.
    """
    if point_color is None:
        point_color = "rgb(30, 20, 160)"

    trace_of_points = go.Scatter3d(
        x=camera[:, 0],
        y=camera[:, 2],
        z=camera[:, 1],
        mode="markers",
        name=name,
        marker=dict(
            symbol="circle",
            size=3,
            color=point_color
        )
    )
    return trace_of_points


def get_gt_figure3d(cameras, points3ds, range_scale=100, connections=CONNECTIONS):
    """
    Generate a Plotly 3D figure for visualization of ground truth camera positions and associated points.

    Args:
        cameras (numpy.ndarray): 3D camera positions to visualize.
        points3ds (list of numpy.ndarray): List of 3D points to visualize.
        range_scale (float, optional): Scaling factor for adjusting the plot range.
        connections (list of list of int, optional): Connections between points.

    Returns:
        plotly.graph_objs.Figure: A Plotly figure object for 3D visualization.
    """
    traces = []

    # Add camera trace
    traces.append(get_camera_trace3d(cameras, BLUE))

    # Add traces for points
    for i, points3d in enumerate(points3ds):
        traces += get_trace3d(points3d, color_string(colors[i]), color_string(colors[i]), str(i),
                              connections=connections)

    layout = go.Layout(
        scene=dict(
            aspectratio=dict(x=1.0, y=1.0, z=1.0),
            xaxis=dict(range=(-0.5 * range_scale, 0.5 * range_scale)),
            yaxis=dict(range=(-0.5 * range_scale, 0.5 * range_scale)),
            zaxis=dict(range=(0, 0.8 * range_scale)),
        ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10)
    )

    return go.Figure(data=traces, layout=layout)


def color_string(color):
    """
    Convert an RGB color represented as a tuple into a string for Plotly color specifications.

    Args:
        color (tuple): RGB color as a tuple of three values (R, G, B) where each value is in the range [0, 1].

    Returns:
        str: A string representation of the RGB color suitable for Plotly.
    """
    return "rgb({}, {}, {})".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
