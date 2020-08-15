#!/usr/bin/env python
# coding: utf-8



"""FAT-GAN"""

__author__ = "Yaohang Li"
__copyright__ = "Copyright 2020, Jefferson Lab and Old Dominion University"
__author__ = "Yaohang Li"
__copyright__ = "Copyright {2020}, {ETHER}"
__version__ = "{1}.{0}"
__maintainer__ = "{maintainer}"
__email__ = "{yaohang@cs.odu.edu}"


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from keras.layers.merge import _Merge, concatenate, dot
from keras.layers.normalization import BatchNormalization
from matplotlib import colors as mcol
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from functools import partial
from keras.models import Model, Sequential, model_from_json
from keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Dropout,
    ActivityRegularization,
    Lambda,
    Concatenate,
    Permute,
    Convolution1D,
    MaxPooling1D,
    AveragePooling1D,
    GlobalAveragePooling1D,
)



# Setting up Environment varialbes
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
IMAGE_DIR_PATH = "gallery/"


# dot product in Minkowski timespace
def vec4dot(v1, v2):
    term0 = v1[0] * v2[0]
    term1 = v1[1] * v2[1]
    term2 = v1[2] * v2[2]
    term3 = v1[3] * v2[3]
    return term0 - term1 - term2 - term3


# four-vector incoming electron momenta
def get_electron_vec():
    beam_electron_vec = [BEAM_ENERGY, 0.0, 0.0, BEAM_ENERGY]
    return beam_electron_vec


# four-vector incoming proton momenta
def get_proton_vec():
    beam_proton_vec = [
        BEAM_ENERGY,
        0.0,
        0.0,
        -np.sqrt(BEAM_ENERGY ** 2.0 - 0.938 * 0.938),
    ]
    return beam_proton_vec


# get proton mass value
def get_mass():
    proton_mass = vec4dot(get_proton_vec(), get_proton_vec())
    return proton_mass


# read and process the pythia event datafile which contains electron
# momenta four-vector for 100,000 scattering electrons
def generate_training_samples(pevent_data_file, BEAM_ENERGY):
    peventnum = 0
    electron = np.empty([1000000, 7])
    with open(pevent_data_file, "r") as fp:
        line = fp.readline()
        while line:

            particle = line.split(",")
            px = float(particle[1])
            py = float(particle[2])
            pz = float(particle[3])
            pxy = px * py
            pxz = px * pz
            pyz = py * pz
            pt = np.sqrt(px * px + py * py)
            e = np.sqrt(px * px + py * py + pz * pz)

            q = [i - j for i, j in zip(get_electron_vec(), [e, px, py, pz])]
            Q2 = -vec4dot(q, q)
            if Q2 > 1.0:
                lnz = np.log(BEAM_ENERGY - pz)
                pzt = pz / pt
                electron[peventnum] = np.array([px, py, lnz, pt, e, pz, pzt])
                peventnum = peventnum + 1
            line = fp.readline()

    # pythia electron momenta four-vector is converted to a feature vector of
    # [px, py, lnz, pt, e, pz, pzt]
    electron = electron[:peventnum]
    return electron


# data for plotting purpose
def generate_electron_features(electron, xyzstd, xyzmean, beam_energy):
    electronDict = {}
    electronDict["Pythia"] = [[], [], [], [], [], [], [], [], [], [], [], []]
    for ii in range(electron.shape[0]):
        pxyz = electron[ii][0:3] * xyzstd + xyzmean
        px = pxyz[0]
        py = pxyz[1]
        lnz = pxyz[2]
        pz = BEAM_ENERGY - np.exp(lnz)
        e = np.sqrt(px * px + py * py + pz * pz)
        pt = np.sqrt(px * px + py * py)
        theta = np.arctan2(pt, pz)
        phi = np.arctan2(py, px)
        pxy = px * py
        pxz = px * pz
        pyz = py * pz
        q = [i - j for i, j in zip(get_electron_vec(), [e, px, py, pz])]
        Q2 = -vec4dot(q, q)
        xbj = Q2 / (2.0 * vec4dot(q, get_proton_vec()))
        electronDict["Pythia"][0].append(e)
        electronDict["Pythia"][1].append(px)
        electronDict["Pythia"][2].append(py)
        electronDict["Pythia"][3].append(pz)
        electronDict["Pythia"][4].append(pt)
        electronDict["Pythia"][5].append(theta)
        electronDict["Pythia"][6].append(phi)
        electronDict["Pythia"][7].append(xbj)
        electronDict["Pythia"][8].append(Q2)
        electronDict["Pythia"][9].append(pxy)
        electronDict["Pythia"][10].append(pxz)
        electronDict["Pythia"][11].append(pyz)

    return electronDict


# MMD loss with Gaussian Kernel
def MMD_loss(x, y):
    sigma = 1
    x1 = x[:HALF_BATCH, :]
    x2 = x[HALF_BATCH:, :]
    y1 = y[:HALF_BATCH, :]
    y2 = y[HALF_BATCH:, :]
    x1_x2 = K.sum(K.exp(-K.sum((x1 - x2) * (x1 - x2),
                               axis=1) / sigma)) / HALF_BATCH
    y1_y2 = K.sum(K.exp(-K.sum((y1 - y2) * (y1 - y2),
                               axis=1) / sigma)) / HALF_BATCH
    x_y = K.sum(K.exp(-K.sum((x - y) * (x - y),
                             axis=1) / sigma)) / BATCH_SIZE

    # 1000 is a tunable weight to balance MMD loss and discriminator loss
    return (x1_x2 + y1_y2 - 2 * x_y) * (x1_x2 + y1_y2 - 2 * x_y) * 1000


# The implementation of the wasserstein_loss
# and gradient penalty loss is based on the wgan-gp example
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate.
    In a standard GAN, the discriminator has a sigmoid output,
    representing the probability that samples are real or generated.
    In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output
    for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated
    samples -1 and real samples 1, instead of the 0 and 1 used in normal GANs,
    so that multiplying the outputs by the labels will give you the loss
    immediately. Note that the nature of this loss means that it can be
    (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred,
                          averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding
    a term to the loss function that penalizes the network if the gradient
    norm moves away from 1. However, it is impossible to evaluate this
    function at all points in the input space. The compromise used in the paper
    is to choose random points on the lines between real and generated samples,
    and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing! In order to evaluate the gradients,
    we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input
    averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.Note that this loss function requires the original averaged
    samples as input, but Keras only supports passing y_true and y_pred to loss
    functions.To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(
        gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))
    )
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms,
    this outputs a random point on the line between each pair of input points.
    """

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


# The generator in GAN only generates [px, py, lnpz]
# Need to reproduce E, pz, pt, pz/pt from the generated features

# Calculate energy from generated features
def energy_mul(x):
    pxyzmean = K.constant(xyzmean)
    pxyzstd = K.constant(xyzstd)
    xyz = x * pxyzstd + pxyzmean
    px = xyz[:, 0:1]
    py = xyz[:, 1:2]
    lnz = xyz[:, 2:3]
    pz = K.constant(BEAM_ENERGY) - K.exp(lnz)
    pxsq = px * px
    pysq = py * py
    pzsq = pz * pz

    energy = (K.sqrt(pxsq + pysq + pzsq) - K.constant(emean))/K.constant(estd)

    return energy


# Calculate pz from generated features
def pz_mul(x):
    pxyzmean = K.constant(xyzmean)
    pxyzstd = K.constant(xyzstd)
    xyz = x * pxyzstd + pxyzmean
    lnz = xyz[:, 2:3]
    pz = K.constant(BEAM_ENERGY) - K.exp(lnz)

    return (pz - K.constant(pzmean)) / K.constant(pzstd)


# Calculate pt from generated features
def pt_mul(x):
    pxyzmean = K.constant(xyzmean)
    pxyzstd = K.constant(xyzstd)
    xyz = x * pxyzstd + pxyzmean
    px = xyz[:, 0:1]
    py = xyz[:, 1:2]
    pxsq = px * px
    pysq = py * py
    pt = (K.sqrt(pxsq + pysq) - K.constant(ptmean)) / K.constant(ptstd)

    return pt


# Calculate pzt from generated features
def pzt_mul(x):
    pxyzmean = K.constant(xyzmean)
    pxyzstd = K.constant(xyzstd)
    xyz = x * pxyzstd + pxyzmean
    px = xyz[:, 0:1]
    py = xyz[:, 1:2]
    lnz = xyz[:, 2:3]
    pz = K.constant(BEAM_ENERGY) - K.exp(lnz)
    pxsq = px * px
    pysq = py * py
    pzt = (pz / K.sqrt(pxsq + pysq) - K.constant(pztmean)) / K.constant(pztstd)

    return pzt


# Define the generator and add energy, pt and pz to the last layer,
# the output is the 4-momentum px, py, pz, E + pz and pt which will
# 6 dimensional vector
def make_generator():
    visible = Input(shape=(100,))
    hidden1 = Dense(512)(visible)
    LR = LeakyReLU(alpha=0.2)(hidden1)
    hidden2 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden2)
    hidden3 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden3)
    hidden4 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden4)
    hidden5 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden5)
    output = Dense(3)(LR)
    energy = Lambda(energy_mul)(output)
    pt = Lambda(pt_mul)(output)
    pz = Lambda(pz_mul)(output)
    pzt = Lambda(pzt_mul)(output)
    outputmerge = concatenate([output, pt, energy, pz, pzt])
    generator = Model(inputs=visible, outputs=[outputmerge])

    return generator


# Define the discriminator and use leakyRelu for all layers and add drop out
def make_discriminator():
    visible = Input(shape=(7,))
    hidden1 = Dense(512)(visible)
    LR = LeakyReLU(alpha=0.2)(hidden1)
    DR = Dropout(rate=0.1)(LR)
    hidden2 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden2)
    DR = Dropout(rate=0.1)(LR)
    hidden3 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden3)
    DR = Dropout(rate=0.1)(LR)
    hidden4 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden4)
    DR = Dropout(rate=0.1)(LR)
    hidden5 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden5)
    DR = Dropout(rate=0.1)(LR)
    output = Dense(1)(DR)
    discriminator = Model(inputs=[visible], outputs=output)

    return discriminator


def make_MMD():
    visible = Input(shape=(7,))
    MMD = Model(inputs=visible, output=visible)

    return MMD


# Plot the features of pythia vs GAN
def plot_GAN_figures(
    electronDict,
    labels,
    binnings,
    titles,
    xbjBins,
    Q2Bins,
    xscales,
    figNames,
    epoch,
    generator,
):
    for i in range(12):
        plt.yscale("log", nonposy="clip")
        plt.xscale(xscales[i], nonposx="clip")
        plt.hist(
            [electronDict[key][i] for key in labels],
            bins=binnings[i],
            histtype="step",
            label=labels,
        )
        plt.title(titles[i])
        plt.legend()
        if figNames[i] == "electronPh":
            plt.gca().set_ylim(bottom=0.01)
        plt.savefig(IMAGE_DIR_PATH + figNames[i] +
                    str(epoch // 100).zfill(5) + ".eps")
        plt.clf()

    # correlation plots
    s = 2.0 * vec4dot(get_electron_vec(), get_proton_vec()) + 0.938 * 0.938

    # Q2-xbj lots
    maxQ2 = [xbjBins, s * xbjBins]
    minQ2 = [xbjBins, [1.0 for i in xbjBins]]
    plt.subplot(121)
    plt.yscale("log", nonposy="clip")
    plt.xscale("log", nonposx="clip")
    plt.hist2d(
        electronDict[labels[0]][7],
        electronDict[labels[0]][8],
        bins=[xbjBins, Q2Bins],
        norm=mcol.LogNorm(),
    )
    plt.plot(maxQ2[0], maxQ2[1], "r-")
    plt.plot(minQ2[0], minQ2[1], "r-")
    plt.title("Q2 vs xBj, pythia")
    plt.ylabel("Q2")
    plt.xlabel("xBj")
    plt.subplot(122)
    plt.yscale("log", nonposy="clip")
    plt.xscale("log", nonposx="clip")
    plt.hist2d(
        electronDict[labels[1]][7],
        electronDict[labels[1]][8],
        bins=[xbjBins, Q2Bins],
        norm=mcol.LogNorm(),
    )
    plt.plot(maxQ2[0], maxQ2[1], "r-")
    plt.plot(minQ2[0], minQ2[1], "r-")
    plt.title("Q2 vs xBj, generated")
    plt.xlabel("xBj")
    plt.savefig(IMAGE_DIR_PATH + "Q2vsxBj" +
                str(epoch // 100).zfill(5) + ".eps")
    plt.clf()

    json_file = "generator" + str(epoch // 100).zfill(5) + ".json"
    generator_json = generator.to_json()
    with open(json_file, "w") as jf:
        jf.write(generator_json)

    generator.save_weights(
        IMAGE_DIR_PATH + "generator" + str(epoch // 100).zfill(5) + ".h5"
    )


# Train the model
def train_FAT_GAN(electronDict):
    Q2min = np.min(electronDict["Pythia"][8])
    generator = make_generator()
    discriminator = make_discriminator()
    MMD = make_MMD()

    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    generator_input = Input(shape=(100,))
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    MMD_Layers_for_generator = MMD(generator_layers)
    generator_model = Model(
        inputs=generator_input,
        outputs=[discriminator_layers_for_generator, MMD_Layers_for_generator],
    )
    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(
        optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
        loss=[wasserstein_loss, MMD_loss],
    )
    generator_model.summary()

    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    real_samples = Input(shape=electron.shape[1:])
    generator_input_for_discriminator = Input(shape=(100,))
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    # We also need to generate weighted-averages of real and generated samples,
    # to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage()(
        [real_samples, generated_samples_for_discriminator]
    )

    # We then run these samples through the discriminator as well.
    # Note that we never really use the discriminator output for these samples,
    # we're only running them to get the gradient norm for the gradient
    # penalty loss.
    averaged_samples_out = discriminator(averaged_samples)

    # The gradient penalty loss function requires the input averaged
    # samples to get gradients. However, Keras loss functions can only have
    # two arguments, y_true and y_pred. We get around this by making
    # a partial() of the function with the averaged samples here.
    partial_gp_loss = partial(
        gradient_penalty_loss,
        averaged_samples=averaged_samples,
        gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT,
    )
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = "gradient_penalty"

    # If we don't concatenate the real and generated samples, however,
    # we get three outputs: One of the generated samples, one of the real
    # samples, and one of the averaged samples, all of size
    # BATCH_SIZE. This works neatly!
    discriminator_model = Model(
        inputs=[real_samples, generator_input_for_discriminator],
        outputs=[
            discriminator_output_from_real_samples,
            discriminator_output_from_generator,
            averaged_samples_out,
        ],
    )
    # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein
    # loss for both the real and generated samples, and the gradient penalty
    # loss for the averaged samples
    discriminator_model.compile(
        optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
        loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
    )
    discriminator_model.summary()

    # We make three label vectors for training. positive_y is the label
    # vector for real samples, with value 1. negative_y is the label vector
    # for generated samples, with value -1. The dummy_y vector is passed to the
    # gradient_penalty loss function and is not used.
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    # Training FAT-GAN for 200,000 epochs
    for epoch in range(200000):
        np.random.shuffle(electron)
        discriminator_loss = []
        generator_loss = []
        minibatches_size = BATCH_SIZE * TRAINING_RATIO
        for i in range(int(electron.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
            discriminator_minibatches = electron[
                i * minibatches_size: (i + 1) * minibatches_size
            ]

            noise = np.random.normal(0, 1, [BATCH_SIZE * TRAINING_RATIO, 100])

            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[
                    j * BATCH_SIZE: (j + 1) * BATCH_SIZE
                ]
                noise_batch = noise[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]

                discriminator_loss.append(
                    discriminator_model.train_on_batch(
                        [image_batch, noise_batch], [positive_y, negative_y, dummy_y]
                    )
                )

            noise = np.random.normal(0, 1, [BATCH_SIZE, 100])
            generator_loss.append(
                generator_model.train_on_batch(noise, [positive_y, image_batch])
            )
        print(epoch, generator_loss)

        # save every 1000 epochs
        if epoch % 1000 == 0:
            print(epoch)
            SAMPLE_SIZE = 200000
            noise = np.random.normal(0, 1, [SAMPLE_SIZE, 100])
            results = generator.predict(noise)

            electronDict["Generated"] = [[], [], [], [], [], [], [], [], [], [], [], []]
            count = 0
            for ii in range(results.shape[0]):
                pxyz = results[ii][0:3] * xyzstd + xyzmean
                px = pxyz[0]
                py = pxyz[1]
                lnz = pxyz[2]
                pz = BEAM_ENERGY - np.exp(lnz)
                pxy = px * py
                pxz = px * pz
                pyz = py * pz
                e = np.sqrt(px * px + py * py + pz * pz)
                pt = np.sqrt(px * px + py * py)
                theta = np.arctan2(pt, pz)
                phi = np.arctan2(py, px)
                q = [i - j for i, j in zip(get_electron_vec(), [e, px, py, pz])]
                Q2 = -vec4dot(q, q)
                xbj = Q2 / (2.0 * vec4dot(q, get_proton_vec()))
                if Q2 > Q2min:
                    count = count + 1
                    electronDict["Generated"][0].append(e)
                    electronDict["Generated"][1].append(px)
                    electronDict["Generated"][2].append(py)
                    electronDict["Generated"][3].append(pz)
                    electronDict["Generated"][4].append(pt)
                    electronDict["Generated"][5].append(theta)
                    electronDict["Generated"][6].append(phi)
                    electronDict["Generated"][7].append(xbj)
                    electronDict["Generated"][8].append(Q2)
                    electronDict["Generated"][9].append(pxy)
                    electronDict["Generated"][10].append(pxz)
                    electronDict["Generated"][11].append(pyz)
                if count > SAMPLE_SIZE / 2:
                    break

            plt.plot(
                electronDict["Generated"][1],
                electronDict["Generated"][2],
                "o",
                markersize=1,
            )
            plt.savefig(IMAGE_DIR_PATH + "hole" + str(epoch // 100).zfill(5) + ".eps")
            plt.clf()

            # generating bins for plotting the graphs
            eBins = np.linspace(0.0, 2.0 * BEAM_ENERGY, 256)
            pBins = np.linspace(-BEAM_ENERGY, BEAM_ENERGY, 56)
            zBins = np.linspace(-BEAM_ENERGY, BEAM_ENERGY + 10, 256)
            xyBins = np.linspace(-BEAM_ENERGY * 10, BEAM_ENERGY * 10, 256)
            xyzBins = np.linspace(-BEAM_ENERGY * 20, BEAM_ENERGY * 20, 256)
            cBins = np.linspace(-5.0, 5.0, 256 * 5)

            thetaBins = np.linspace(0.0, 1.0 * 3.142, 100)
            phiBins = np.linspace(-3.142, 3.142, 100)
            xbjBins = np.logspace(-3.0, 0.0, 100)
            Q2Bins = np.logspace(0.0, 4.0, 100)

            binnings = [
                eBins,
                pBins,
                pBins,
                zBins,
                pBins,
                thetaBins,
                phiBins,
                xbjBins,
                Q2Bins,
                xyBins,
                xyzBins,
                xyzBins,
            ]
            figNames = [
                "electronEn",
                "electronPx",
                "electronPy",
                "electronPz",
                "electronPt",
                "electronTh",
                "electronPh",
                "xBjorken",
                "Qsquared",
                "pxy",
                "pxz",
                "pyz",
            ]
            titles = [
                "Electron Energy",
                "Electron Px",
                "Electron Py",
                "Electron Pz",
                "Electron Pt",
                "Electron Theta",
                "Electron Phi",
                "x-Bjorken",
                "Q-Squared",
                "pxy",
                "pxz",
                "pyz",
            ]
            xscales = [
                "linear",
                "linear",
                "linear",
                "linear",
                "linear",
                "linear",
                "linear",
                "log",
                "log",
                "linear",
                "linear",
                "linear",
            ]

            plot_GAN_figures(
                electronDict,
                labels,
                binnings,
                titles,
                xbjBins,
                Q2Bins,
                xscales,
                figNames,
                epoch,
                generator,
            )

# data for plotting purpose
labels = ["Pythia", "Generated"]

# pythia data path
pevent_data_file = "data/tape2.txt"

# Hard code BEAM ENERGIES, Assume equal energy
BEAM_ENERGY = 50.0

# HALF_BATCH and FULL_BATCH SIZES
HALF_BATCH = 8000
BATCH_SIZE = HALF_BATCH * 2

# The training ratio is the number of discriminator updates
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10
electron = generate_training_samples(pevent_data_file, BEAM_ENERGY)

# Normalization
electronmean = np.mean(electron, axis=0)
electronstd = np.std(electron, axis=0)
electron = (electron - electronmean) / electronstd
xyzmean = electronmean[0:3]
xyzstd = electronstd[0:3]
ptmean = electronmean[3]
ptstd = electronstd[3]
emean = electronmean[4]
estd = electronstd[4]
pzmean = electronmean[5]
pzstd = electronstd[5]
pztmean = electronmean[6]
pztstd = electronstd[6]
electronDict = generate_electron_features(electron, xyzstd, xyzmean, BEAM_ENERGY)

# Train FAT-GAN model
def main():

    train_FAT_GAN(electronDict)

main()
