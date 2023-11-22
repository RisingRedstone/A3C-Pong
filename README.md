# Asynchronous Advantage Actor-Critic (A3C) Network with TensorFlow

Welcome to the Asynchronous Advantage Actor-Critic (A3C) implementation using TensorFlow! This project utilizes the A3C algorithm to train an agent to play the Pong game from the OpenAI Gym environment. Read on to learn about the A3C algorithm, how to use this implementation, and the available usage methods.

## A3C Algorithm

### Introduction

The Asynchronous Advantage Actor-Critic (A3C) algorithm is a reinforcement learning technique that combines policy gradient methods with value function approximation. It uses multiple agents running in parallel to explore different parts of the environment and update the policy and value functions asynchronously.

## Usage

To get started, use the following command to view available options:

```bash
python3 train.py --help
```

### Training

To train the A3C network, use the following command:

```bash
python3 train.py --n_workers {No of workers, default: 1, recommended: 16} PongNoFrameskip-v4
```

### Testing

To test an existing network, use the following command:

```bash
python3 train.py --testing True PongNoFrameskip-v4 --savedir {Saved Directory + /network}
```

Example:

```bash
python3 train.py --testing True PongNoFrameskip-v4 --savedir Saves/Save2/network
```

### Save During Training

To save a network while training, press 's'.

## Contributing

If you'd like to contribute, please fork the repository and create a pull request. Feel free to open issues for feature requests or bug reports.

Feel free to explore, experiment, and enhance the A3C implementation. Happy training!
