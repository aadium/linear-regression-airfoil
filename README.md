# Linear Regression Breakdown

I have implemented an algorithm that uses gradient descent to train a linear regression model. I have implemented functions for calculating loss and gradient, calculating the model weights (a process which is called training), and evaluating the model's performance.

### Dataset Description

The [NASA 'airfoil' dataset](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise#) was "obtained from a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic [soundproof] wind tunnel." - R. Lopez, PhD

An airfoil is the cross-sectional shape of a wing, blade or propeller. For this particular dataset, specific variables were measured that are known to contribute to the amount of noise the airfoil generates when exposed to smooth air flow. For more information about the prediction of noise production from airfoils, check out this [link](https://ntrs.nasa.gov/citations/19890016302).

Inputs: 
- Frequency, in Hz
- Angle of attack, in degrees
- Chord length [length of airfoil], in meters
- Free-stream velocity, in meters per second
- Suction side displacement thickness, in meters

Output: 
- Scaled sound pressure level, in decibels.

Preview:

| Pressure (dB) | Frequency (Hz) | Angle (degrees) | Length  (m) | Velocity (m/s) | Displacement (m) 
|--|--|--|--|--|--|
| 126.201 | 800 | 0 | 0.3048 | 71.3 | 0.002663 
| 125.201 | 1000| 0 | 0.3048 | 71.3 | 0.002663 
| 125.951 | 1250| 0 | 0.3048 | 71.3 | 0.002663 