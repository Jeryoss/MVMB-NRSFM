import numpy as np
from scipy.spatial.distance import cosine


def euclidean_distance(landmarks1, landmarks2):

    """Calculates the Euclidean distance between two sets of landmarks.

    Args:
        landmarks1: A numpy array of landmarks.
        landmarks2: A numpy array of landmarks.

    Returns:
        A numpy array of Euclidean distances between corresponding landmarks.
    """

    # Ensure both sets of landmarks have the same length
    if len(landmarks1) != len(landmarks2):
        raise ValueError("Landmark sets must have the same length.")

    # Calculate the Euclidean distance between corresponding landmarks
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)

    return distances


def compute_similarity(landmarks1, landmarks2):

    """Calculates the cosine similarity between two sets of landmarks.

    Args:
        landmarks1: A numpy array of landmarks.
        landmarks2: A numpy array of landmarks.

    Returns:
        A numpy array of cosine similarities between corresponding landmarks.
    """

    # Ensure both sets of landmarks have the same length
    if len(landmarks1) != len(landmarks2):
        raise ValueError("Landmark sets must have the same length.")

    # Reshape landmark arrays for compatibility with cosine function
    landmarks1_flat = landmarks1.flatten()
    landmarks2_flat = landmarks2.flatten()

    # Calculate cosine similarity
    similarity = 1 - cosine(landmarks1_flat, landmarks2_flat)

    return similarity