# TensorGeodesic
Ray tracing of a Schwarzschild black hole written entirely in TensorFlow.


## Dependencies:

- Python 3

- TensorFlow 2.x

- numpy

- matplotlib

## About
The Schwarzschild metric is currently implemented, with Christoffel symbols computed via automatic differentiation by tf.GradientTape.batch_jacobian. Geodesics from the observer are integrated using the classic fourth Runge-Kutta method (RK4).

Since the program is written in TensorFlow, it is fully differentiable and supports arbitrary CPU/GPU systems. In particular, it works on my M1 Max MacBook Pro.


Tested on TensorFlow 2.7.0, with Apple's tensorflow-metal 0.3.0 plugin (!!!)
