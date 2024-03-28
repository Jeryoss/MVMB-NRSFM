import cv2
import os


def get_numeric_part(filename):
    """Extracts the numeric part from a filename.

    The numeric part is the part of the filename that comes before the file extension.
    For example, for the filename "0.png", the numeric part is "0".

    Args:
        filename: The filename to extract the numeric part from.

    Returns:
        The numeric part of the filename.
    """

    try:
        # Extract the numeric part by removing the file extension and converting to an integer
        return int(os.path.splitext(filename)[0])
    except ValueError:
        # If the filename doesn't contain a valid number, return a large value to push it to the end
        return float('inf')


def create_video(image_folder, output_path, frame_rate):
    """Creates a video from a folder of images.

    The images in the folder are sorted by their numeric part, and then converted to a video using the specified frame rate.

    Args:
        image_folder: The folder containing the images to be converted to a video.
        output_path: The path to the output video file.
        frame_rate: The frame rate of the output video.
    """

    # Get a list of image files (e.g., '0.png', '1.png', etc.)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    if not image_files:
        print("No image files found in the specified folder.")
        return

    # Sort the image files based on the numeric part of their names
    image_files = sorted(image_files, key=get_numeric_part)

    # Read the first image to get dimensions
    sample_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, layers = sample_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)
    out.release()
    print("Video creation complete.")
