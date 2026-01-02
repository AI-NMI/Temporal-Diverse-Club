# Temporal-Diverse-Club

This repository contains the implementation and datasets for the "Temporal-Diverse-Club" analysis. We provide three datasets of varying scales, along with the original raw data. This codebase allows for the replication of our findings regarding temporal network dynamics, covering diverse metrics from participation coefficients to complex club transitions.

## Overview

The repository is structured to analyze three distinct types of network data. We summarize the contributions and distinctions of the analyses performed in the following modules:

### 1. Neuroscience Analysis
**File:** `neuroscience.py`

This module focuses on neurological network data and implements the following analyses:
* **Participation Coefficient:** Analysis of node integration across modules.
* **Clubness:** Measurement of the "rich-club" phenomenon or similar structural groupings.
* **Machine Learning:** Application of ML models for pattern recognition in neural data.
* **Edge Density:** (Detailed in Supplementary Materials) Analysis of connectivity density changes.

### 2. Airline Network Analysis
**File:** `airline.py`

This module processes temporal flight data to understand transport dynamics:
* **Temporal Participation Coefficient:** extending the standard metric to time-varying networks.
* **Clubness:** Identification of central hubs over time.
* **Traditional Metrics:** Calculation of three standard network science metrics for baseline comparison.
* **Kuramoto Model:** Simulation of synchronization dynamics on the airline network.
* **Clique Detection:** (Detailed in Supplementary Materials) Identification of fully connected subgraphs.

### 3. Ant Colony Analysis
**File:** `ant.py`

This module analyzes social insect interaction networks:
* **Weighted Participation Coefficient:** Accounting for interaction strength in community integration.
* **Community Alignment:** An algorithm to align community labels across temporal snapshots.
* **Strong vs. Weak Networks:** Comparative analysis of tie strengths.
* **Club Transition:** Tracking how the composition of the "club" changes over time.

## Data Structure & Parameters

This framework handles temporal data using specific naming conventions. **Please pay close attention to the distinction between uppercase and lowercase variables.**

### Key Variables
* **`dy_mat`**: `np.ndarray` (Real Matrices)
    * Shape: $T \times N \times N$ (Time $\times$ Nodes $\times$ Nodes)
* **`dy_gra`**: (Real Graphs)
    * Shape: $T \times G$ (Time $\times$ Graph Objects)
* **`dy_ran`**: `np.ndarray` (Null Network)
    * Shape: $T \times N \times N$
* **`agg`**: Aggregated Matrix (Weighted sum or mean of temporal layers)
* **`AGG`**: Aggregated Graph object

> **Note:** In this repository, `G` (Uppercase) explicitly denotes a **Graph object**, while `g` (Lowercase) generally denotes a **matrix** representation.

## Installation

### Prerequisites

To run the code in this repository, you will need the following dependencies:

* **NetworkX** 3.3
* **Pandas** 2.2.3
* **Community** 1.1.0 (python-louvain)
* **Infomap** 1.0.1

You can install the required packages using pip:

```bash
pip install networkx==3.3 pandas==2.2.3 community==1.1.0 infomap==1.0.1

