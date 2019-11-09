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
    imageio.imwrite(
        "bayes_net/{0}_output.jpg".format(filename.split(".")[0].split("/")[-1]),
        draw_edge(
            image=image,
            y_coordinates=edge_strength_matrix.argmax(axis=0),
            color=(255, 123, 0),
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
    normally_distributed_initial_probabilities = np.full(
        edge_strength_matrix.shape[0], 1 / edge_strength_matrix.shape[0]
    )
    max_of_each_row = np.amax(edge_strength_matrix, axis=0)
    emission_probabilities = np.array(
        [
            each_row / max_of_each_row[index]
            for index, each_row in enumerate(edge_strength_matrix)
        ]
    )

    all_rows = range(edge_strength.shape[0])
    column_value_for_each_row = [row for row in all_rows]

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

    first_product, second_product = (
        np.empty((transitional_probabilities.shape[0], edge_strength_matrix.shape[1])),
        np.empty((transitional_probabilities.shape[0], edge_strength_matrix.shape[1])),
    )

    first_product[:, 0] = (
        normally_distributed_initial_probabilities * emission_probabilities[:, 0]
    )
    second_product[:, 0] = 0

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

    resultant_matrix = np.empty(edge_strength_matrix.shape[1])
    resultant_matrix[-1] = np.argmax(
        first_product[:, edge_strength_matrix.shape[1] - 1]
    )

    for column in reversed(range(1, edge_strength_matrix.shape[1])):
        resultant_matrix[column - 1] = second_product[
            int(resultant_matrix[column]), column
        ]

    imageio.imwrite(
        "viterbi/{0}_output.jpg".format(filename.split(".")[0].split("/")[-1]),
        draw_edge(
            image=image,
            y_coordinates=resultant_matrix,
            color=(100, 10, 130),
            thickness=5,
        ),
    )

    return None


def human_viterbi(edge_strength_matrix: np.array, filename: str, image, coordinates: list = None) -> None:
    normally_distributed_initial_probabilities = np.full(
        edge_strength_matrix.shape[0], 1 / edge_strength_matrix.shape[0]
    )
    max_of_each_row = np.amax(edge_strength_matrix, axis=0)

    # TODO take 10 pixels below and above for emission
    emission_probabilities = np.array(
        [
            each_row / max_of_each_row[index]
            for index, each_row in enumerate(edge_strength_matrix)
        ]
    )

    return None


if __name__ == "__main__":

    if not os.path.exists("viterbi") and not os.path.exists("bayes_net"):
        os.makedirs("viterbi")
        os.makedirs("bayes_net")

    (input_filename, gt_row, gt_col) = sys.argv[1:]

    # load in image
    input_image_bayes_net = Image.open(input_filename)
    input_image_viterbi = Image.open(input_filename)

    # compute edge strength mask
    edge_strength = get_edge_strength(input_image_bayes_net)
    imageio.imwrite(
        "edges.jpg", np.uint8(255 * edge_strength / (np.amax(edge_strength)))
    )

    bayes_net(
        edge_strength_matrix=edge_strength,
        filename=input_filename,
        image=input_image_bayes_net,
    )

    viterbi(
        edge_strength_matrix=edge_strength,
        filename=input_filename,
        image=input_image_viterbi,
    )

    # TODO Human Viterbi
    human_viterbi(
        edge_strength_matrix=edge_strength,
        filename=input_filename,
        image=input_image_viterbi,
    )
