# GAN for CICIDS-2017 Dataset

This project implements a Generative Adversarial Network (GAN) to generate synthetic data based on the CICIDS-2017 dataset. The GAN consists of a generator and a discriminator, which are trained together to produce realistic synthetic data.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction

Generative Adversarial Networks (GANs) are a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014. GANs consist of two neural networks, a generator and a discriminator, that compete against each other in a game. The generator tries to produce realistic data, while the discriminator tries to distinguish between real and fake data.

This project uses a GAN to generate synthetic data based on the CICIDS-2017 dataset, which is a comprehensive dataset for network intrusion detection.

## Requirements

- Python 3.7+
- TensorFlow 2.0+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/gan-cicids2017.git
    cd gan-cicids2017
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the CICIDS-2017 dataset:
    - Download the dataset from [here](https://www.unb.ca/cic/datasets/ids-2017.html).
    - Place the CSV files in a directory, e.g., `/path/to/CICIDS2017`.

2. Update the file path in the `main` function in `gan_cicids2017.py`:
    ```python
    file_path = '/path/to/CICIDS2017/yourfile.csv'
    ```

3. Run the script:
    ```sh
    python3 gan_cicids2017.py
    ```

## Results

The script will generate synthetic data and save the following:
- Generated samples at regular intervals during training.
- Training losses for both the generator and discriminator.

### Example Output

Generated samples and loss plots will be saved in the `gan_samples` directory.



