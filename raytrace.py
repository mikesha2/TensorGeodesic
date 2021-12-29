#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 02:48:25 2021

@author: mikesha
"""

import tensorflow as tf
from tensorflow import norm, sqrt, atan2, sin, cos, abs, reduce_sum as sum, \
    zeros, einsum, stack, concat, zeros_like, ones
from tensorflow.linalg import diag, inv
import numpy as np
from numpy import indices
#tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.set_floatx('float64')
#%%
resolution = 100

# Schwarzschild radius 2GM/c^2
@tf.function
def schwarzschildMetric(positions, rs=1):
    # Start with metric in Schwarzschild coordinates
    rSquared = sum(positions * positions, axis=-1) + 1e-6
    r = sqrt(rSquared)
    m00 = -(1-rs/r)
    m11 = -1 / m00
    cosTheta = positions[..., -1] / r
    sinSquaredTheta = abs(1 - cosTheta * cosTheta)
    m22 = r * r
    m33 = m22 * sinSquaredTheta
    g = diag(stack((m11, m22, m33), axis=-1))
    
    sinTheta = sqrt(sinSquaredTheta)
    phi = atan2(positions[..., 1], positions[..., 0])
    
    # r = sqrt(x^2+y^2+z^2)
    # dr = x dx / r
    j0x = positions / r[..., None]
    xySquared = sum(positions[..., :-1] * positions[..., :-1], axis=-1) + 1e-6
    xyNorm = sqrt(xySquared)
    
    # theta = arctan(sqrt(x^2+y^2)/z)
    # dtheta = (xz / [sqrt(x^2+y^2) r^2], yz / [sqrt(x^2+y^2) r^2], -sqrt(x^2+y^2) / r^2)
    factor = positions[..., 2] / (xyNorm * rSquared)
    j10 = positions[..., 0] * factor
    j11 = positions[..., 1] * factor
    j12 = -xyNorm / rSquared
    j1x = stack((j10, j11, j12), axis=-1)
    
    # phi = arctan2(y/x) (adds a constant due to the branch cut, derivative wipes out constant)
    # dphi = (-y / (x^2+y^2), x / (x^2 + y^2), 0)
    j20 = -positions[..., 1] / xySquared
    j21 = positions[..., 0] / xySquared
    j22 = zeros_like(positions[..., 0])
    j2x = stack((j20, j21, j22), axis=-1)
    jacobian = stack((j0x, j1x, j2x), axis=-2)
    
    # Change of basis for metric from spherical to cartesian -> g_ab dr^a / dx^i dr^b / dx^j
    g = einsum('...ab,...ai,...bj->...ij', g, jacobian, jacobian)
    
    # there are no mixed 2-forms between time and space for Schwarzschild coordinates
    return g, inv(g)

@tf.function
def dMetric(positions, metric=schwarzschildMetric, rs=1):
    with tf.GradientTape() as t:
        g, g_inv = metric(positions, rs=rs)
        batch = (np.prod(g.shape[:-2]), 3, 3)
        dg = t.batch_jacobian(tf.reshape(g, batch), tf.reshape(positions, batch[:-1]))
        dg = tf.reshape(dg, (*g.shape, 3))
    return g, g_inv, dg

@tf.function
def christoffel(positions, metric=schwarzschildMetric, rs=1):
    g, g_inv, dg = dMetric(positions, metric=metric, rs=rs)
    Gamma = 0.5 * (tf.einsum('...im,...mkl->...ikl', g_inv, dg) + \
                   tf.einsum('...im,...mlk->...ikl', g_inv, dg) - \
                   tf.einsum('...im,...klm->...ikl', g_inv, dg))
    return g, g_inv, dg, Gamma

@tf.function
def geodesicCoeffs(positions, velocities, metric, rs):
    _, _, _, Gamma = christoffel(positions, metric=metric, rs=rs)
    Gamma.trainable = False
    return -tf.einsum('...mab,...a,...b->...m', Gamma, velocities, velocities)
    

# We have to evaluate the Christoffel symbols 4 times at each step
@tf.function
def RK4(positions, velocities, metric=schwarzschildMetric, rs=-1, dt=1e-2):
    velocities = (velocities / norm(velocities, axis=-1, keepdims=True))
    k11 = dt * velocities
    k21 = dt * geodesicCoeffs(positions, velocities, metric, rs)
    
    velocities21 = velocities + 0.5 * k21
    
    k12 = dt * velocities21
    k22 = dt * geodesicCoeffs(positions + 0.5 * k11, velocities21, metric, rs)
    
    velocities22 = velocities + 0.5 * k22
    k13 = dt * velocities22
    k23 = dt * geodesicCoeffs(positions + 0.5 * k12, velocities22, metric, rs)
    
    velocities23 = velocities + k23
    k14 = dt * velocities23
    k24 = dt * geodesicCoeffs(positions + k13, velocities23, metric, rs)
    
    positions = positions + ((k11 + 2 * (k12 + k13) + k14) / 6)
    velocities = velocities + ((k21 + 2 * (k22 + k23) + k24) / 6)
    return positions, velocities

#%% Assume observer is facing y axis, black hole at origin

# Helper function to perform raycasting so that TensorFlow can compile/optimize the graph
def raycast(background, positions, velocities, clip, maxIters, resolution, metric, rs):
    result = np.zeros(resolution)
    terminated = tf.Variable(tf.zeros(resolution, dtype=tf.bool), trainable=False)
    
    limit = 2.5 * rs
    
    for i in range(maxIters):
        print(i)
        if i % 10 == 0:

            numTerminated = tf.reduce_sum(tf.cast(terminated, tf.uint16))
            print(numTerminated.numpy(), positions[0][0].numpy(), velocities[0][0].numpy())
            if  numTerminated > 9 * resolution[0] * resolution[1] // 10:
                break
        for i in range(3):
            positions, velocities = RK4(positions, velocities, metric=metric, rs=rs)
        # Unstable photons are terminated
        previouslyTerminated = tf.identity(terminated)
        terminated = (tf.logical_or(tf.logical_or(terminated, (tf.reduce_sum(positions * positions, axis=-1) < limit)), tf.reduce_sum(positions[..., 1]) > 5))
        result += tf.where(tf.logical_and(tf.logical_not(previouslyTerminated), terminated), background(positions[..., 0], positions[..., 2]), 0)
    return result

def runRaycast(background=None, screenDistance=2, blackHoleDistance=1, clip=3, resolution=(1920, 1080), fov=90, maxIters=5000, metric=schwarzschildMetric, rs=1):
    pixelLength = screenDistance * np.arctan(fov / 2) / (resolution[0] / 2)
    positions = np.zeros((*resolution, 3))
    totalDist = blackHoleDistance + screenDistance
    positions[..., 1] = -totalDist
    positions = tf.Variable(positions, trainable=True)
    center = (*[i // 2 for i in resolution],)
    
    # Set up initial velocities for raycasting
    velocities = indices(resolution).astype(np.float64)
    velocities[0, ...] -= center[0]
    velocities[1, ...] -= center[1]
    velocities *= pixelLength
    velocities = np.stack((velocities[0], screenDistance * np.ones(resolution), velocities[1]), axis=-1)
    velocities /= np.linalg.norm(velocities, axis=-1, keepdims=True)
    velocities = tf.Variable(velocities, trainable=False)
    
    # Set up a background if there is none
    if background is None:
        background = lambda x,y: tf.exp(-((x-1)*(x-1) + (y+2)*(y+2)) / 2) / 6.283185307179586

    return raycast(background, positions, velocities, clip, maxIters, resolution, metric, rs)

result = runRaycast()

import matplotlib.pyplot as plt
plt.matshow(result)

