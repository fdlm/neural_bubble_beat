import copy
import os
import pickle
from itertools import product

import begin
import cv2
import madmom as mm
import matplotlib.pyplot as plt
import numpy as np
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import LogarithmicFilteredSpectrogramProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

# physical constants
mu0 = 4.0 * np.pi * (10 ** (-7))  # vacuum permeability
c = 299792458  # speed of light
epsilon0 = 1.0/(mu0 * c**2)  # vacuum permittivity

# visualization colors for positive and negative charge
pos_color = '#BC252B'
neg_color = '#4B88A2'

# parameters for adjusting the influence of the gravity and the
# electrostatic force
l_gravity = 1e-2
l_coulomb = 1e-6

# background properties
video_size = 32


def gauss_2d(mu, sigma, n_samples):
    x = np.random.normal(mu, sigma, n_samples)
    y = np.random.normal(mu, sigma, n_samples)
    return np.asarray((x, y)).transpose()


def compute_force_field(cluster, qs, inverse=False):
    Q = np.dot(qs, qs.transpose())
    D = squareform(pdist(cluster))
    D = np.clip(D, 0.1, np.inf)
    denom = np.power(D, 0.5) * 4 * np.pi * epsilon0
    vec = np.zeros((D.shape[0], D.shape[1], 2))

    fnc = lambda x,y: x-y
    vec[:, :, 0] = squareform(pdist(cluster[:, 0:1], metric=fnc))
    vec[:, :, 1] = squareform(pdist(cluster[:, 1:2], metric=fnc))

    if inverse:
        vec[:, :, 0] = -np.tril(vec[:, :, 0]) + np.triu(vec[:, :, 0])
        vec[:, :, 1] = -np.tril(vec[:, :, 1]) + np.triu(vec[:, :, 1])
    else:
        vec[:, :, 0] = np.tril(vec[:, :, 0]) - np.triu(vec[:, :, 0])
        vec[:, :, 1] = np.tril(vec[:, :, 1]) - np.triu(vec[:, :, 1])

    R = (vec / D[:, :, np.newaxis])
    F = (Q / denom)[:, :, np.newaxis] * R
    F = np.nan_to_num(F)
    return F.sum(0)


def downbeat_activations(audio_file):
    # keep only last network
    dbc = mm.features.downbeats.RNNDownBeatProcessor()
    dbc.processors = dbc.processors[:-1]
    dbc.processors[-1].processors[0].processors = \
    dbc.processors[-1].processors[0].processors[:1]
    nn = dbc.processors[-1].processors[0].processors[0]

    # get hidden layer output
    nn.layers = nn.layers[:-2]
    return dbc(audio_file)


def drum_activations(audio_file):
    class PadProcessor:
        def __init__(self, pad):
            self.pad = pad

        def __call__(self, data):
            pad_start = np.repeat(data[:1], self.pad, axis=0)
            pad_stop = np.repeat(data[-1:], self.pad, axis=0)
            return np.concatenate((pad_start, data, pad_stop))

    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    frames = FramedSignalProcessor(frame_size=2048, fps=100)
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    spec = LogarithmicFilteredSpectrogramProcessor(
        num_channels=1, sample_rate=44100,
        filterbank=LogarithmicFilterbank, frame_size=2048, fps=100,
        num_bands=12, fmin=30, fmax=15000,
        norm_filters=True)

    preproc = SequentialProcessor(
        [sig, frames, stft, spec, PadProcessor(pad=12)])

    nn = pickle.load(open('drums_cnn0_O8_S2.pkl'))
    nn.layers = nn.layers[:-8]

    output = nn(preproc(audio_file))
    output = output.reshape(output.shape[0], -1)
    return output


def harmonic_background(audio_file, fps=100):
    dc = mm.audio.chroma.DeepChromaProcessor()
    output = dc(audio_file)
    hue, sat = PCA(2).fit_transform(output).T

    hue = np.interp(np.linspace(0, len(hue), len(hue) * (fps / 10)),
                    np.arange(len(hue)), hue)
    sat = np.interp(np.linspace(0, len(sat), len(sat) * (fps / 10)),
                    np.arange(len(sat)), sat)

    hue -= hue.min()
    hue /= hue.max()
    hue *= 180

    sat -= sat.min()
    sat /= sat.max()
    sat *= 255

    val_mask = np.ones((video_size, video_size))
    idxs = np.array(list(product(range(video_size), range(video_size))))
    c = np.array([video_size / 2, video_size / 2])
    for i in range(len(idxs)):
        val_mask[tuple(idxs[i])] = np.sqrt(sum((idxs[i] - c) ** 2))
    val_mask -= val_mask.min()
    val_mask /= val_mask.max()

    drum_val = drum_background(audio_file)

    blank = np.ones((video_size, video_size))

    frames = []
    for i, (h, s, v) in enumerate(zip(hue, sat, drum_val)):
        frame_hue = blank * h
        frame_sat = blank * s
        frame_val = v * val_mask
        frame = np.stack([frame_hue, frame_sat, frame_val], axis=-1).astype(
            np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
        frames.append(frame)

    return frames


def drum_background(audio_file):
    output = drum_activations(audio_file)
    z = (output - output.mean(0, keepdims=True)) / \
        np.maximum(0.0001, output.std(0, keepdims=True))
    clusters = KMeans(5).fit(abs(z).T).predict(abs(z).T)
    exclude_cluster = np.bincount(clusters).argmax()

    c0 = np.array([0, 0])
    c1 = np.array([video_size - 1, 0])
    c2 = np.array([0, video_size - 1])
    c3 = np.array([video_size - 1, video_size - 1])
    idxs = np.array(list(product(range(video_size), range(video_size))))

    c_idxs = list()

    c_idxs.append(
        [(x, y) for x, y in idxs[np.sum((idxs - c0) ** 2, 1).argsort()]])
    c_idxs.append(
        [(x, y) for x, y in idxs[np.sum((idxs - c1) ** 2, 1).argsort()]])
    c_idxs.append(
        [(x, y) for x, y in idxs[np.sum((idxs - c2) ** 2, 1).argsort()]])
    c_idxs.append(
        [(x, y) for x, y in idxs[np.sum((idxs - c3) ** 2, 1).argsort()]])
    for i in range(4):
        c_idxs[i].remove(tuple(c0))
        c_idxs[i].remove(tuple(c1))
        c_idxs[i].remove(tuple(c2))
        c_idxs[i].remove(tuple(c3))

    cluster_map = {
        i: j for i, j in zip(set(range(5)) - set([exclude_cluster]), range(4))}

    val = np.zeros((len(output), video_size, video_size))
    for i, c in enumerate(clusters):
        if c == exclude_cluster:
            continue
        c = cluster_map[c]
        neuron_idx = c_idxs[c][0]
        for j in range(4):
            c_idxs[j].remove(neuron_idx)
        val[:, neuron_idx[0], neuron_idx[1]] = output[:, i]

    val -= val.min()
    val /= val.max()
    val *= 160

    return val


@begin.start
def main(audio_file, write_frames=False, inverse=False):

    # load pre calculated activations and subtract the mean
    activations = downbeat_activations(audio_file)
    activations -= np.mean(activations, 0, keepdims=True)

    n_particles = activations.shape[1]

    # create particles according to a gaussian with mean 0 and unit variance
    particles = gauss_2d(0, 1, n_particles)

    # coforce vector field with momentum
    F_mom = np.zeros((n_particles, 2))

    # particle_history
    particle_history = [[] for _ in range(n_particles)]
    max_history = 5

    fig = plt.figure(figsize=(10, 10))
    plt.style.use('dark_background')
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    background_frames = harmonic_background(audio_file)

    if write_frames and not os.path.exists('figs'):
        os.mkdir('figs')

    n_frames = len(activations)
    frame_data = zip(activations, background_frames)

    for i, (act, bg_img) in tqdm(enumerate(frame_data), total=n_frames):
        # the activations are used as the electric charge for the coulomb law
        qs = act[:, np.newaxis] * l_coulomb
        # calculate the electrostatic force field
        F_coulomb = compute_force_field(particles, qs, inverse=inverse)
        # add a gravity force that draws all points towards the center
        F_grav = particles * l_gravity
        # add up forces
        F = F_coulomb - F_grav
        # add momentum to the forces
        F_mom = 0.9 * F_mom + 0.1 * F
        # determine particles with negative/positive charge
        neg = act < 0
        pos = act >= 0

        # plot/store every fifth frame (will result in a frame rate of 20 fps)
        if i % 5 == 0:
            fig.clf()

            plt.imshow(bg_img, extent=(-0.3, 0.3, -0.3, 0.3))

            plt.xlim([-0.3, 0.3])
            plt.ylim([-0.3, 0.3])

            plt.scatter(particles[neg, 0], particles[neg, 1],
                        c=neg_color, s=np.abs(act[neg]) * 1000 + 10)
            plt.scatter(particles[pos, 0], particles[pos, 1],
                        c=pos_color, s=np.abs(act[pos]) * 1000 + 10)

            # plot the particle history
            for index, hist in enumerate(particle_history):
                d = np.abs(act[index]) * 1000 + 10
                color = neg_color if act[index] < 0 else pos_color
                for h_idx, h in enumerate(hist):
                    size = d - (d / max_history) * (h_idx + 1)
                    plt.scatter(h[0], h[1], c=color, s=size,
                                alpha=1 - (h_idx + 1) * (1.0 / max_history))

            # store latest particle in history and clip the history
            for j, p in enumerate(particles):
                particle_history[j].insert(0, copy.copy(p))
                if len(particle_history[j]) >= max_history:
                    particle_history[j] = particle_history[j][0:-1]

            plt.draw()

            if write_frames:
                plt.savefig('figs/%05d.png' % (i/5))
            else:
                plt.pause(0.01)

        # adjust particle positions
        particles += F_mom
