# Simulation Code for "Probably Approximately Correct Off-Policy Prediction of Contextual Bandits"

This repository contains the simulation code for Section 4 (Synthetic Data Experiments) of our paper.

The code evaluates and compares the performance of our proposed algorithm **PACOPP** with two comparison algorithms, **COPP** and **COPP-RS**, in terms of the **coverage** and **average length** of their predictive intervals.

The experiments are conducted using **Python 3.13.1**, and we provide a **single-threaded** implementation.  

Due to the high computational cost of the simulation, we recommend enabling **parallelization** or reducing the number of simulation repetitions to speed up execution.  

For reference, the experimental results have already been provided in the folder `results_of_comparative_experiment/`.

## How to Run

1. Install the required Python packages.

2. Run the main script:

   ```bash
   python main_OPE_comparative.py

3. The experimental results will be saved in the folder:
   ```bash
   results_of_comparative_experiment/
