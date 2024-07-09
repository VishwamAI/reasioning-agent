# Reasoning Agent Advanced

## Introduction

This project aims to develop a reasoning agent that surpasses the capabilities of GPT Gemini and Anthropic's Claude AI. The primary focus is on leveraging reinforcement learning to achieve a 100% pass rate in various reasoning tasks. Recent enhancements include the addition of a conversational interface and the development of the agent as a user bot capable of answering questions.

## Purpose

The purpose of this project is to create an advanced reasoning agent that can effectively model and reason about complex environments and interactions. By incorporating state-of-the-art techniques in recursive reasoning and probabilistic modeling, the agent will be able to make informed decisions and adapt to changing scenarios.

## Project Structure

- `src/`: Contains the source code for the reasoning agent.
  - `agent.py`: Implementation of the reasoning agent.
  - `train_agent.py`: Script for training the reasoning agent and starting interactive sessions.
- `tests/`: Contains the test cases for the reasoning agent.
  - `test_agent.py`: Tests for the reasoning agent.
- `docs/`: Contains the documentation for the project.
  - `README.md`: Project overview and documentation.
  - `evaluation_criteria.md`: Outlines the evaluation criteria for the reasoning agent.
  - `high_level_architecture.md`: Describes the high-level architecture of the reasoning agent.
  - `project_timeline.md`: Provides a detailed project timeline.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- TensorFlow
- OpenAI Gym (optional, for reinforcement learning environment)
- spaCy (for natural language processing)

### Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/VishwamAI/reasoning-agent.git
   cd reasoning-agent
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv reasoning-agent-venv
   source reasoning-agent-venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install tensorflow gym spacy
   ```

4. Install the spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Development

### High-Level Architecture

The reasoning agent will be designed based on the following key components:
- **Recursive Reasoning Graph (R2G)**: Models the recursive reasoning process of agents.
- **Probabilistic Recursive Reasoning (PR2)**: Uses probabilistic modeling to approximate opponents' conditional policies.

### Core Components

- **Agent**: The main reasoning agent that interacts with the environment and learns from it.
- **Environment**: The simulated environment in which the agent operates and learns.
- **Learning Algorithms**: The reinforcement learning algorithms used to train the agent.

### Current Status

The project is currently focused on training the reasoning agent using advancement datasets and developing the agent as a user bot capable of answering questions. The agent has been enhanced with a conversational interface and can handle a variety of queries related to its state, training, and decision-making process.

## Usage

### Training the Agent

To train the reasoning agent, run the `train_agent.py` script with the desired parameters:
```bash
python src/train_agent.py --mode train --env_name CartPole-v1 --episodes 1000 --batch_size 32
```

### Interactive Session

To start an interactive session with the trained agent, run the `train_agent.py` script in interactive mode:
```bash
python src/train_agent.py --mode interactive --env_name CartPole-v1
```

During the interactive session, you can ask the agent questions about its state, training progress, and more.

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.
