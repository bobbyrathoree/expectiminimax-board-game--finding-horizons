#!/usr/local/bin/python3
#
# Authors: Bobby Rathore (brathore), Neha Supe (nehasupe), Kelly Wheeler (kellwhee)
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2019

from PIL import Image
import numpy as np
from scipy.ndimage import filters
import sys
import os
import imageio


class HMM:
    def __init__(self, transitional_probabilities, emission_probabilities):
        """
        Constructor for the HMM model class
        :param transitional_probabilities:
        :param emission_probabilities:
        """
        self.transitional_probabilities = transitional_probabilities
        self.emission_probabilities = emission_probabilities

    def get_emission_dist(self, emission):
        return self.emission_probabilities[:, emission]

    @property
    def get_number_of_states(self):
        return self.transitional_probabilities.shape[0]

    @property
    def get_transition_probabilities(self):
        return self.transitional_probabilities


def get_edge_strength(input_image) -> np.array:
    """
    calculate "Edge strength map" of an image
    :param input_image: the original image for which we want the edge strengths
    :return: a 2D numpy vector containing the edge strength matrix
    """
    grayscale = np.array(input_image.convert("L"))
    filtered_y = np.zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    return np.sqrt(filtered_y ** 2)


def draw_edge(image, y_coordinates, color, thickness):
    """
    draw a "line" on an image (actually just plot the given y-coordinates for each x-coordinate)
    image is the image to draw on y_coordinates is a list, containing the
    y-coordinates and length equal to the x dimension size of the image
    color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
    thickness is thickness of line in pixels
    :param image: the original image object to put pixels on
    :param y_coordinates: the coordinates where we shall put pixels
    :param color: color of the pixels
    :param thickness: density/thickness of the pixels
    :return: image object with superimposed pixels
    """
    for (x, y) in enumerate(y_coordinates):
        for t in range(
            int(max(y - int(thickness / 2), 0)),
            int(min(y + int(thickness / 2), image.size[1] - 1)),
        ):
            image.putpixel((x, t), color)
    return image


def bayes_net(edge_strength_matrix: np.array, filename: str, image) -> None:
    """
    Naive Bayes Net implementation for the given image matrix
    :param edge_strength_matrix: a 2D numpy vector containing the edge strength matrix
    :param filename: name of the file we're working with
    :param image: the original image object to put pixels on
    :return: None
    """
    # Save for our records
    imageio.imwrite(
        "bayes_net/{0}_output_simple.jpg".format(filename.split(".")[0].split("/")[-1]),
        draw_edge(
            image=image.copy(),
            y_coordinates=edge_strength_matrix.argmax(axis=0),
            color=(0, 0, 255),
            thickness=5,
        ),
    )

    # Save for grading system
    imageio.imwrite(
        "output_simple.jpg",
        draw_edge(
            image=image,
            y_coordinates=edge_strength_matrix.argmax(axis=0),
            color=(0, 0, 255),
            thickness=5,
        ),
    )
    return None


def viterbi(edge_strength_matrix: np.array, filename: str, image) -> None:
    """
    Viterbi implementation for the given image matrix
    :param edge_strength_matrix: a 2D numpy vector containing the edge strength matrix
    :param filename: name of file we're working with
    :param image: the original image object to put pixels on
    :return: None
    """
    # Initial probabilities are normally distributed
    normally_distributed_initial_probabilities = np.full(
        edge_strength_matrix.shape[0], 1 / edge_strength_matrix.shape[0]
    )
    max_of_each_row = np.amax(edge_strength_matrix, axis=0)

    # Setup Emission Probabilities
    emission_probabilities = np.array(
        [
            each_row / max_of_each_row[index]
            for index, each_row in enumerate(edge_strength_matrix)
        ]
    )

    # Setup transitional Probabilities
    all_rows = range(edge_strength_matrix.shape[0])
    column_value_for_each_row = [row for row in all_rows]

    # Take absolute value of each row normalized with it's corresponding column value
    transitional_probabilities = np.array(
        [
            np.array(
                [
                    abs(
                        (
                            column
                            - np.max(
                                [
                                    abs(row - absolute_column_value)
                                    for absolute_column_value in column_value_for_each_row
                                ]
                            )
                        )
                        / np.max(
                            [
                                abs(row - absolute_column_value)
                                for absolute_column_value in column_value_for_each_row
                            ]
                        )
                    )
                    for column in [
                        abs(row - absolute_column_value)
                        for absolute_column_value in column_value_for_each_row
                    ]
                ]
            )
            for row in all_rows
        ]
    )

    # Viterbi algorithm begins here
    first_product, second_product = (
        np.empty((transitional_probabilities.shape[0], edge_strength_matrix.shape[1])),
        np.empty((transitional_probabilities.shape[0], edge_strength_matrix.shape[1])),
    )

    # Setup the first column by multiplying the initial and emission probabilities
    first_product[:, 0] = (
        normally_distributed_initial_probabilities * emission_probabilities[:, 0]
    )
    second_product[:, 0] = 0

    # Update each column along the way with the maximum value
    for column in range(1, edge_strength_matrix.shape[1]):
        first_product[:, column] = np.max(
            first_product[:, column - 1]
            * transitional_probabilities.T
            * emission_probabilities[:, column].T,
            1,
        )
        second_product[:, column] = np.argmax(
            first_product[:, column - 1] * transitional_probabilities.T, 1
        )

    # Get the y_coordinates ready
    resultant_matrix = np.empty(edge_strength_matrix.shape[1])
    resultant_matrix[-1] = np.argmax(
        first_product[:, edge_strength_matrix.shape[1] - 1]
    )

    # "Backtracking"
    for column in reversed(range(1, edge_strength_matrix.shape[1])):
        resultant_matrix[column - 1] = second_product[
            int(resultant_matrix[column]), column
        ]

    # Save for our records
    imageio.imwrite(
        "viterbi/{0}_output_map.jpg".format(filename.split(".")[0].split("/")[-1]),
        draw_edge(
            image=image.copy(),
            y_coordinates=resultant_matrix,
            color=(255, 0, 0),
            thickness=5,
        ),
    )

    # Save for grading system
    imageio.imwrite(
        "output_map.jpg",
        draw_edge(
            image=image, y_coordinates=resultant_matrix, color=(255, 0, 0), thickness=5
        ),
    )

    return None


def viterbi_with_logs(
    edge_strength_matrix: np.array,
    filename: str,
    image,
    row: int = 0,
    column: int = 0,
    human: bool = False,
) -> None:
    """
    Similar viterbi implementation as above but without initial probabilities,
    with human feedback incorporation and logs to avoid underflow errors
    :param edge_strength_matrix: a 2D numpy vector containing the edge strength matrix
    :param filename: the name of the file we're working with
    :param image: the original image object to put pixels on
    :param row: row pixel value where human thinks the ridge is
    :param column: column pixel value where human thinks the ridge is
    :param human: a flag to determine whether human feedback is to be considered or not
    :return: None
    """
    # Setup Transitional Probabilities
    transition_probabilities = np.repeat(
        np.linspace(
            0, edge_strength_matrix.shape[0] - 1, edge_strength_matrix.shape[0]
        ),
        edge_strength_matrix.shape[0],
    ).reshape(
        edge_strength_matrix.shape[0], edge_strength_matrix.shape[0]
    ).T - np.linspace(
        0, edge_strength_matrix.shape[0] - 1, edge_strength_matrix.shape[0]
    ).reshape(
        edge_strength_matrix.shape[0], 1
    )

    # Take the max of each column for each row and reshape
    transition_probabilities = np.amax(
        np.absolute(transition_probabilities), axis=0
    ).reshape(1, edge_strength_matrix.shape[0]) - np.absolute(transition_probabilities)

    # Divide by sum to normalize
    transition_probabilities = transition_probabilities / np.sum(
        transition_probabilities, axis=0
    ).reshape(1, edge_strength_matrix.shape[0])

    # Add a small value to avoid log infinity errors
    transition_probabilities += 0.00001

    # Setup Emission probabilities
    emission_probabilities = edge_strength_matrix / np.amax(
        edge_strength_matrix, axis=0
    )

    # Add a small value to avoid log infinity errors
    emission_probabilities += 0.00001

    # If we're considering human feedback in our algorithm,
    # Make the column values essentially equal to zero
    # And make that particular pixel equal to 1
    if human:
        emission_probabilities[:, column] = 0.00000001
        emission_probabilities[row, column] = 1

    # All observable intermediate states
    intermediate_states = np.array([0 for _ in range(edge_strength_matrix.shape[0])])
    all_possible_paths = [str(path) for path in range(edge_strength_matrix.shape[0])]

    # Make a backup to act as a backtracking device
    intermediate_states_backup = intermediate_states.copy()

    for column in range(1, edge_strength_matrix.shape[1]):
        for row in range(edge_strength_matrix.shape[0]):
            intermediate_states[row] = np.amax(
                np.log(transition_probabilities[:, row])
                + intermediate_states_backup
                + np.log(emission_probabilities[row, column])
            )
            all_possible_paths[row] += " " + np.argmax(
                np.log(transition_probabilities[:, row])
                + intermediate_states_backup
                + np.log(emission_probabilities[row, column])
            ).astype(str)
        intermediate_states_backup = intermediate_states.copy()

    resultant_matrix = np.array(
        all_possible_paths[np.argmax(intermediate_states)].split(" ")
    ).astype(int)

    # Save for our records
    imageio.imwrite(
        "{0}/{1}_output_{2}.jpg".format(
            "human_viterbi" if human else "viterbi",
            filename.split(".")[0].split("/")[-1],
            "human" if human else "map",
        ),
        draw_edge(
            image=image, y_coordinates=resultant_matrix, color=(0, 255, 0), thickness=5
        ),
    )

    # Save for grading system
    imageio.imwrite(
        "output_human.jpg",
        draw_edge(
            image=image, y_coordinates=resultant_matrix, color=(0, 255, 0), thickness=5
        ),
    )

    return None


if __name__ == "__main__":

    if not os.path.exists("viterbi"):
        os.makedirs("viterbi")

    if not os.path.exists("bayes_net"):
        os.makedirs("bayes_net")

    if not os.path.exists("human_viterbi"):
        os.makedirs("human_viterbi")

    (input_filename, gt_row, gt_col) = sys.argv[1:]

    # load in image
    input_image_bayes_net = Image.open(input_filename)
    input_image_viterbi = Image.open(input_filename)
    input_image_viterbi_with_log = Image.open(input_filename)

    # compute edge strength mask
    edge_strength = get_edge_strength(input_image_bayes_net)
    imageio.imwrite(
        "edges.jpg", np.uint8(255 * edge_strength / (np.amax(edge_strength)))
    )

    # Module 1: Bayes Net
    bayes_net(
        edge_strength_matrix=edge_strength,
        filename=input_filename,
        image=input_image_bayes_net,
    )

    # Module 2: Viterbi
    viterbi(
        edge_strength_matrix=edge_strength,
        filename=input_filename,
        image=input_image_viterbi,
    )

    # Module 3: Viterbi with human feedback
    viterbi_with_logs(
        edge_strength_matrix=edge_strength,
        filename=input_filename,
        image=input_image_viterbi_with_log,
        row=int(gt_row),
        column=int(gt_col),
        human=True,
    )
