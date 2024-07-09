import gym
from agent import ReasoningAgent
import argparse

def train_agent(env_name, episodes, batch_size):
    try:
        agent = ReasoningAgent(env_name)
        agent.train(episodes, batch_size)
        return agent
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None

def run_example_queries(agent):
    if agent:
        print(agent.handle_query("What is the current state?"))
        print(agent.handle_query("What is the total reward?"))
        print(agent.handle_query("Is the episode done?"))
        print(agent.handle_query("What action should be taken?"))
        print(agent.handle_query("How many episodes have been trained?"))
        print(agent.handle_query("What is the agent's learning rate?"))
        print(agent.handle_query("Tell me about the agent's model architecture."))
        print(agent.handle_query("What are the agent's hyperparameters?"))
        print(agent.handle_query("What is the batch size?"))
        print(agent.handle_query("Hello"))

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
    parser = argparse.ArgumentParser(description="Train a reasoning agent.")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="Name of the gym environment.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    args = parser.parse_args()

    # Train the agent
    trained_agent = train_agent(args.env_name, args.episodes, args.batch_size)

    # Run example queries
    run_example_queries(trained_agent)

    # Start interactive session
    interactive_session(trained_agent)
