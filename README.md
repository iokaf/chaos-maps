# chaos-maps

## Project Description

This module aims in automating the numerical simulation and study of discrete
 chaotic maps. 
More explicitly, the class DiscreteMap allows the definition of a discrete chaotic
maps based on thier iterative step for 1- and n- dimensional systems.
Additionaly, given parameter values and initial conditions, system trajectories 
can be generated.
Additionaly, numerical approximation of the Lyapunov exponent for each for the 
system's variables can be performed.
The method for estimating the Lyapunov exponent utilizes the QR approximation 
method, as described in [1].

Additionally, the plotting submodule introduces the ChaoticMapPlot class.
The constructor of this class takes a ChaoticMap object as input and provides
tools for creating plots to analyze the source map's dynamical behavior.
Currently, these tools include the bifurcation diagrams for a given initial condition
and parameter set, the Lyapunov exponent diagram, the return map and Cobweb diagram.

## Installing - Getting started

The chaotic-maps library can be installed using pip through
> pip install chaos-maps

The GitHub repository for the package is


## Developing

This module was built with Python 3.9 but requires ^3.6.

The non-standard libraries required for its creation are 

- numpy
- matplotlib

Installing the libary installs these versions of the modules, if the requirement for their existence is not satisfied.

## Configuration
The plt_config.py in the *plotting* directory contains the default values for creating matplotlib diagrams.

These values can be changed in two ways:

1. Altering the default values by changing teh respective fields in the configuration file.

2. Pass the desired named arguments for the plots when calling
the method. This way the desired changes are applied to the particular figure only, while preserving the default values intact.

## Tests
Tests for the correctness of the outputs in terms of consistency have been implemented.
It is suggested that the tests are run at least upon installation of the library, in order to verify the correctness of the installation.
The list of tests shall extend with future versions.

## Licencing
Copyright 2022 Dr. Ioannis Kafetzis

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## References

[1] He J., Yu S., & Cai J. (2016). Numerical analysis and improved algorithms for Lyapunov-exponent calculation of discrete-time chaotic systems. International Journal of Bifurcation and Chaos, 26(13), 1650219.