## Overview

This repository provides the official implementation and datasets for the study "Temporal Diverse Club Phenomenon in Complex Dynamical Systems," introducing the **Temporal Diverse Club (TDC)** as a novel topological framework to quantify the integrative backbone of evolving networks. Distinct from the Temporal Rich Club (TRC), which consolidates local resources through high-strength connections, the TDC identifies nodes that persistently maintain diverse cross-community bridges over time, acting as critical mediators for global information flow and system adaptability. Our multidisciplinary analysis verifies this principle across three distinct scales: in **human brain networks** (microscopic), we reveal that TDC nodes are centrally anchored in the higher-order cognitive systems and exhibit a "pathological rigidity" in Schizophrenia patients; in the **US air transportation network** (macroscopic), TDC airports function as national gateways that facilitate system-wide synchronization significantly faster than regional TRC hubs; and in **ant colony social networks** (mesoscopic), we uncover a "social maturation" mechanism where individuals transition from early-phase, exploratory TDC roles driven by weak ties to mature, consolidated TRC leadership positions. By integrating these findings, this codebase offers a unified toolkit, including algorithms for randomization strategy and clubness calculation, to explore how the temporal persistence of functional architecture, rather than static topology, governs the resilience and evolutionary trajectories of complex systems.

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

