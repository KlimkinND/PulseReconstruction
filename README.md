## All-optical pulse reconstruction with machine learning

In this repository we provide the code necessary to reproduce the results of the paper "Deep neural networks for high harmonic spectroscopy in solids" (Klimkin, Ivanov (2020)), currently pending approval.

The code is divided into three files. The first one, named **`resp-2h-gpu.jl`**, serves to generate the training set by simulating the high harmonic responses of multiple tight-binding crystals to strong IR field using TDSE in the single active electron approximation (see paper for details), with the parameters defined within the code. 
It generates a file named **`resps-cep-[ddmm_HHMM].jld2`**, where ddmm_HHMM designates the date and time. 

The neural network, defined in **`nn-cep.jl`**, takes in a JLD2 file generated by **`resp-2h-gpu.jl`**, with the label set using the `prefix` parameter, then augments the data (see paper), and trains itself to retrieve the pulse parameters and, optionally, band parameters, from the response data.
This code, in its turn, generates a file named **`results-[ddmm_HHMM].jld2`**.

After this, the visualization is done with **`fig.jl`**. This file uses the responses (for the simulation parameters) and the corresponding neural network recognition results to visually compare the true and reconstructed dynamics and draw the reconstruction figure from the paper.