from src.generate_embeddings import (
    train_model,
    get_embedding,
    train_model_with_data,
    get_embedding_with_data,
)
from src.MNIST import embed, load_data, predict_on_embedded
import ott
from matplotlib import pyplot as plt, colormaps

from jax import nn, numpy as jnp, vmap, random
import numpy as np


def run_experiment(
    rng,
    hidden_layer_sizes,
    embedding_layer=2,
    with_oracle=False,
    verbose=False,
    num_training_epochs=3,
):
    rngs = random.split(rng, 2)
    print() if verbose else None
    print("---------------------------") if verbose else None
    print("Loading Data")
    train_data, test_data, num_pixels, num_labels = load_data()

    print("Training model 1") if verbose else None
    params_1 = train_model_with_data(
        hidden_layer_sizes,
        train_data,
        test_data,
        num_pixels,
        num_labels,
        key=rngs[0],
        verbose=False,
        num_epochs=num_training_epochs,
    )
    print("Training model 2") if verbose else None
    params_2 = train_model_with_data(
        hidden_layer_sizes,
        train_data,
        test_data,
        num_pixels,
        num_labels,
        key=rngs[1],
        verbose=False,
        num_epochs=num_training_epochs,
    )

    print("Getting embeddings") if verbose else None
    (
        train_embedding_1,
        test_embedding_1,
        train_labels_1,
        test_labels_1,
    ) = get_embedding_with_data(1_000, params_1, embedding_layer, train_data, test_data)
    (
        train_embedding_2,
        test_embedding_2,
        train_labels_2,
        test_labels_2,
    ) = get_embedding_with_data(1_000, params_2, embedding_layer, train_data, test_data)

    print("Computing OT") if verbose else None
    geom_1 = ott.geometry.pointcloud.PointCloud(test_embedding_1, scale_cost="mean")
    geom_2 = ott.geometry.pointcloud.PointCloud(test_embedding_2, scale_cost="mean")
    prob = ott.problems.quadratic.quadratic_problem.QuadraticProblem(
        geom_1, geom_2, scale_cost=True
    )
    solver = ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein(epsilon=0.005)
    soln = solver(prob, max_iterations=10_000)

    def cross_predict(embedded_1):
        probs = nn.softmax(
            train_embedding_1 @ embedded_1 / jnp.linalg.norm(train_embedding_1)
        )
        cross_probs = soln.matrix.T @ probs
        cross_probs /= cross_probs.sum()
        if with_oracle:
            return (train_labels_2 * cross_probs[:, None]).sum(0)
        cross_embedding = train_embedding_2.T @ cross_probs
        cross_prediction = predict_on_embedded(
            params_2, cross_embedding, embedding_layer
        )
        return cross_prediction

    batched_cross_predict = vmap(cross_predict)

    test_pred = batched_cross_predict(test_embedding_1)
    test_accuracy = (test_pred.argmax(1) == test_labels_1.argmax(1)).mean()
    print(f"test accuracy: {test_accuracy:.3f}") if verbose else None
    return test_accuracy
