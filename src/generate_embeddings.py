from src.MNIST import load_data, init_network_params, loss, accuracy, batched_embed
import optax
from jax import random, jit, grad, value_and_grad
import time
import numpy as np


def get_train_batches(batch_size, train_data):
    train_images = np.array(train_data.images)
    train_labels = np.array(train_data.labels)
    num_train = train_images.shape[0]
    idx = np.arange(num_train)
    np.random.shuffle(idx)
    for i in range(0, num_train, batch_size):
        batch_idx = idx[i : i + batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]


def train_model(
    hidden_layer_sizes,
    key=None,
    verbose=True,
    num_epochs=10,
    batch_size=128,
    learning_rate=1e-3,
):
    # Load data
    train_data, test_data, num_pixels, num_labels = load_data()
    return train_model_with_data(
        hidden_layer_sizes,
        train_data,
        test_data,
        num_pixels,
        num_labels,
        key,
        verbose,
        num_epochs,
        batch_size,
        learning_rate,
    )


def train_model_with_data(
    hidden_layer_sizes,
    train_data,
    test_data,
    num_pixels,
    num_labels,
    key=None,
    verbose=True,
    num_epochs=10,
    batch_size=128,
    learning_rate=1e-3,
):
    layer_sizes = [num_pixels] + hidden_layer_sizes + [num_labels]
    print("Loaded data") if verbose else None

    if key is None:
        key = random.PRNGKey(0)
        print("No key provided, using default key") if verbose else None

    # Initialize parameters
    init_params = init_network_params(key, layer_sizes)
    print("Initialized parameters") if verbose else None

    # Train
    num_batches = train_data.images.shape[0] // batch_size

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)
    params = init_params
    print("Initialized optimizer") if verbose else None

    @jit
    def step(params, opt_state, x, y):
        loss_val, grads = value_and_grad(loss)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in get_train_batches(batch_size, train_data):
            params, opt_state, loss_val = step(params, opt_state, x, y)
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, train_data.images, train_data.labels)
        test_acc = accuracy(params, test_data.images, test_data.labels)
        if verbose:
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))
            print()

    return params


def stratified_subsamples(images, labels, num_samples_per_class):
    num_classes = 10
    num_samples = num_classes * num_samples_per_class
    subsample_images = np.zeros((num_samples, images.shape[1]))
    subsample_labels = np.zeros((num_samples, labels.shape[1]))
    for i in range(num_classes):
        idx = np.where(labels[:, i] == 1)[0]
        np.random.shuffle(idx)
        subsample_images[
            i * num_samples_per_class : (i + 1) * num_samples_per_class
        ] = images[idx[:num_samples_per_class]]
        subsample_labels[
            i * num_samples_per_class : (i + 1) * num_samples_per_class
        ] = labels[idx[:num_samples_per_class]]
    return subsample_images, subsample_labels


def get_embedding(data_size, params, layer_number):
    train_data, test_data, num_pixels, num_labels = load_data()
    return get_embedding_with_data(
        data_size, params, layer_number, train_data, test_data
    )


def get_embedding_with_data(data_size, params, layer_number, train_data, test_data):
    samples_per_class = data_size // 10
    train_images, train_labels = stratified_subsamples(
        train_data.images, train_data.labels, samples_per_class
    )
    test_images, test_labels = stratified_subsamples(
        test_data.images, test_data.labels, samples_per_class
    )
    train_embed = batched_embed(params, train_images, layer_number)
    test_embed = batched_embed(params, test_images, layer_number)
    return train_embed, test_embed, train_labels, test_labels
