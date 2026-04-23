# Amygdala Sleep Dynamics Analysis Pipeline

This repository contains the electrophysiological data analysis pipeline developed as part of the doctoral thesis: "Neuronal dynamics underlying homeostasis in the basolateral-amygdala during sleep".
Scientific Overview

This codebase is designed to process and analyze large-scale recordings (LFP and single-unit activity) from the basolateral amygdala (BLA) in rats. The pipeline characterizes neuronal dynamics and network properties across different vigilance states: WAKE, NREM sleep, and REM sleep.

## Key findings supported by this pipeline include:

    Increased activity and neural desynchronization specific to the BLA during REM sleep.

    A homeostatic regulation process where firing rates increase during wakefulness and decrease during NREM sleep.

## Core Features
1. Firing Rate Dynamics

    State-Specific FR: Automatic calculation of mean firing rates for principal neurons and interneurons across all brain states.

    REM-ON Identification: Statistical identification of neurons with activity specifically correlated with REM sleep.

    Transition Dynamics: Analysis of firing rate changes at state boundaries (e.g., NREM to REM).

    Homeostatic Tracking: Measurement of firing rate evolution during Extended Wake (EWE) or Extended Sleep Episodes (ESE).

2. Network Metrics

    Excitatory-Inhibitory Balance (EIB): Calculation of the activity ratio between pyramidal and interneuron populations.

    Coefficient of Variation (CV): Measurement of firing rate variability and dispersion within the neuronal population.

    Synchrony: Evaluation of pairwise correlation used as a proxy for global network synchronicity.


## Technical Usage and Dependencies

The project requires standard scientific libraries (numpy, scipy) as well as the specialized neuroseries (https://github.com/NeuroNetMem/neuroseries) library.
Documentation

Author: Billel Khouader, MD-PhD student
Affiliation: Institut du Fer à Moulin, Inserm U1270, Sorbonne Université
