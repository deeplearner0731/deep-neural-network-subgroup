# Deep Learning-Based Ranking Method for Subgroup and Predictive Biomarkers Identification

## Overview

**DeepRAB** is a deep learning-based framework designed for identifying subgroups and predictive biomarkers in precision medicine. 

## Features
- **DeepRAB Framework**: Implements the core DeepRAB model for identifying subgroups and predictive biomarkers.
- **Causal Forest Framework**: Integrates the Causal Forest (CF) model for estimating conditional average treatment effects (CATE) as a comparison.
- **XGBoost with Modified Loss Function**: A customized version of XGBoost tailored for biomarker identification, incorporating an A-learning loss function.
- **Linear Regression Models**: Implements linear regression with both modified outcomes and modified covariates.

## Latest Update — July 2025
We’ve refactored the DeepRAB codebase to address previously reported bugs and improve stability. Additionally, we’ve added a demo script for hyperparameter tuning. For demonstration purposes, the script includes only a small set of hyperparameters. Users are encouraged to explore a broader hyperparameter space when running on HPC environments.
## Latest Update (01/2026)
We added demo code demonstrating simple hyperparameter tuning for the continuous case.

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- R 4.0+




