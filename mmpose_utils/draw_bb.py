import cv2
import numpy as np
import os


def draw_bounding_boxes(frames, bounding_boxes_list, output_folder, colors, index=0):
    """Draws bounding boxes on the given frames and saves the results to the output folder.

    Args:
        frames: A list of frames to draw bounding boxes on.
        bounding_boxes_list: A list of lists of bounding boxes. Each inner list corresponds to a frame in the `frames` list.
        output_folder: The folder to save the results to.
        colors: A list of colors to use for the bounding boxes.
        index: The index of the current video. This is used to create unique filenames for the saved frames.
    """

    # # Check if the number of frames and bounding box lists match
    # if len(frames) != len(bounding_boxes_list):
    #     raise ValueError("Number of frames and bounding box lists must be the same.")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Merge frames horizontally with bounding boxes and save the result
    merged_frame_horizontal = merge_frames_with_bounding_boxes(frames, bounding_boxes_list, colors)
    merged_horizontal_filename = os.path.join(output_folder, f"merged_horizontal_{index}.png")
    save_frame(merged_frame_horizontal, merged_horizontal_filename)


def save_frame(frame, filename):
    """Saves the given frame as an image file.

    Args:
        frame: The frame to save.
        filename: The filename to save the frame to.
    """

    resized_frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
    cv2.imwrite(filename, cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))


def merge_frames_with_bounding_boxes(frames, bounding_boxes_list, colors):
    """Merges frames horizontally with bounding boxes.

    Args:
        frames: A list of frames to merge.
        bounding_boxes_list: A list of lists of bounding boxes. Each inner list corresponds to a frame in the `frames` list.
        colors: A list of colors to use for the bounding boxes.

    Returns:
        The merged frame with bounding boxes.
    """

    frames_with_bboxes = []
    for frame, bounding_boxes in zip(frames, bounding_boxes_list):
        # Draw bounding boxes on the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for j, bbox in enumerate(bounding_boxes):
            color = colors[j]  # Get a unique color for each bounding box
            size = j + 4
            draw_single_box(frame_rgb, bbox, color, size=size)
            # break
        frames_with_bboxes.append(frame_rgb)
    # Concatenate frames horizontally
    return np.concatenate(frames_with_bboxes, axis=1)


def get_color(index):
    """Generates a unique color based on the index.

    Args:
        index: The index to generate the color for.

    Returns:
        A list of RGB values for the color.
    """

    np.random.seed(index + 10)
    return list(np.random.random(size=3) * 255)


def draw_single_box(frame, bbox, color, size=4):
    """Draws the bounding box on the frame.

    Args:
        frame: The frame to draw the bounding box on.
        bbox: The bounding box to draw.
        color: The color to use for the bounding box.
    """

    x1, y1, x2, y2 = convert_boxes_to_int(bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, size)


def merge_frames(frames, axis=0):
    """Merges frames horizontally (axis=1) or vertically (axis=0).

    Args:
        frames: A list of frames to merge.
        axis: The axis to merge the frames along.

    Returns:
        The merged frame.
    """

    return np.concatenate(frames, axis=axis)


def convert_boxes_to_int(bounding_boxes_list):
    """Converts decimal bounding box values to integers.

    Args:
        bounding_boxes_list: A list of lists of bounding boxes. Each inner list corresponds to a frame in the `frames` list.

    Returns:
        A list of lists of bounding boxes, where each inner list contains integers.
    """

    return [np.round(boxes).astype(int) for boxes in bounding_boxes_list]


def xywh_to_xyxy(bounding_boxes_list):
    """Converts (x, y, width, height) to (x1, y1, x2, y2).

    Args:
        bounding_boxes_list: A list of lists of bounding boxes. Each inner list corresponds to a frame in the `frames` list.

    Returns:
        A list of lists of bounding boxes, where each inner list contains (x1, y1, x2, y2) values.
    """

    return [np.column_stack((boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3])) for boxes
            in bounding_boxes_list]


if __name__ == '__main__':
    # Example usage:
    # frames is a list of frames (each frame is a numpy array)
    # bounding_boxes_list is a list of lists of bounding boxes corresponding to each frame
    # Each bounding box is represented as (x, y, width, height)

    # Example frames and bounding boxes (you should replace these with your data)
    frames = [np.zeros((300, 400, 3), dtype=np.uint8) for _ in range(5)]
    bounding_boxes_list = [
        [(50, 50, 100, 100), (150, 80, 120, 150)],
        [(100, 80, 120, 150), (200, 30, 80, 120)],
        [(30, 20, 200, 100), (100, 150, 120, 80)],
        [(50, 30, 80, 120), (180, 70, 100, 120)],
        [(10, 10, 150, 150), (70, 100, 120, 100), (200, 50, 80, 80)]
    ]

    # Draw bounding boxes on frames
    draw_bounding_boxes(frames, bounding_boxes_list)
