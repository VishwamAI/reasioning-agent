import gym
from agent import ReasoningAgent
import argparse
import logging

def train_agent(env_name, episodes, batch_size, dataset_path=None):
    try:
        agent = ReasoningAgent(env_name)
        if dataset_path:
            # Load the dataset and integrate it into the training process
            logging.info(f"Loading dataset from {dataset_path}")
            # Add dataset loading logic here
        agent.train(episodes, batch_size)
        return agent
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        return None

def run_example_queries(agent):
    if agent:
        logging.info(agent.handle_query("What is the current state?"))
        logging.info(agent.handle_query("What is the total reward?"))
        logging.info(agent.handle_query("Is the episode done?"))
        logging.info(agent.handle_query("What action should be taken?"))
        logging.info(agent.handle_query("How many episodes have been trained?"))
        logging.info(agent.handle_query("What is the agent's learning rate?"))
        logging.info(agent.handle_query("Tell me about the agent's model architecture."))
        logging.info(agent.handle_query("What are the agent's hyperparameters?"))
        logging.info(agent.handle_query("What is the batch size?"))
        logging.info(agent.handle_query("Hello"))

def interactive_session(agent):
    if agent:
        print("Interactive session started. Type 'exit' to end the session.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = agent.handle_query(user_input)
            print(f"Agent: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reasoning agent or start an interactive session.")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="Name of the gym environment.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--dataset_path", type=str, help="Path to the advancement dataset.")
    parser.add_argument("--mode", type=str, choices=["train", "interactive"], default="interactive", help="Mode to run the script in: 'train' or 'interactive'.")
    args = parser.parse_args()

    if args.mode == "train":
        # Train the agent
        trained_agent = train_agent(args.env_name, args.episodes, args.batch_size, args.dataset_path)
        # Run example queries
        run_example_queries(trained_agent)
        # Start interactive session
        interactive_session(trained_agent)
    else:
        # Initialize the agent without training
        agent = ReasoningAgent(args.env_name)
        # Start interactive session
        interactive_session(agent)
