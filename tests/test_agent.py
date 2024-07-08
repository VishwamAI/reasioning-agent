import unittest
from src.agent import ReasoningAgent

class TestReasoningAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.agent = ReasoningAgent(env_name="CartPole-v1")
        # Commenting out the training process to speed up the test suite
        # cls.agent.train(episodes=10, batch_size=32)
        cls.agent.episodes_trained = 10  # Set episodes_trained to a non-zero value for testing
        cls.agent.done = True  # Set done to True for testing

    def test_handle_query_state(self):
        response = self.agent.handle_query("What is the current state?")
        self.assertIn("The current state is:", response)

    def test_handle_query_reward(self):
        response = self.agent.handle_query("What is the total reward?")
        self.assertIn("The total reward accumulated is:", response)

    def test_handle_query_done(self):
        self.assertTrue(self.agent.done)  # Verify the state before querying
        response = self.agent.handle_query("Is the episode done?")
        self.assertIn("The episode is", response)
        self.assertIn("done", response)
        response = self.agent.handle_query("Is the episode complete?")
        self.assertIn("The episode is", response)
        self.assertIn("done", response)
        response = self.agent.handle_query("Is the episode finished?")
        self.assertIn("The episode is", response)
        self.assertIn("done", response)

    def test_handle_query_completion_status(self):
        self.assertTrue(self.agent.done)  # Verify the state before querying
        response = self.agent.handle_query("What is the episode completion status?")
        self.assertIn("The episode is", response)
        self.assertIn("done", response)

    def test_handle_query_action(self):
        response = self.agent.handle_query("What is the chosen action?")
        self.assertIn("The chosen action is:", response)

    def test_handle_query_progress(self):
        self.assertEqual(self.agent.episodes_trained, 10)  # Verify the state before querying
        response = self.agent.handle_query("What is the progress?")
        self.assertIn("The agent has been trained for", response)
        self.assertIn("episodes", response)

    def test_handle_query_decision(self):
        response = self.agent.handle_query("What is the decision-making process?")
        self.assertIn("The agent's decision-making process involves", response)

    def test_handle_query_training(self):
        self.assertEqual(self.agent.episodes_trained, 10)  # Verify the state before querying
        response = self.agent.handle_query("What is the learning status?")
        self.assertIn("The agent has been trained for", response)
        self.assertIn("episodes", response)

    def test_handle_query_hyperparameters(self):
        response = self.agent.handle_query("What are the hyperparameters?")
        self.assertIn("The agent's hyperparameters are:", response)

    def test_handle_query_architecture(self):
        response = self.agent.handle_query("What is the model architecture?")
        self.assertIn("The agent's model architecture consists of", response)
        if self.agent.episodes_trained == 0:
            self.assertIn("Dense layers with ReLU activations", response)
        else:
            self.assertIn("Dense layers with ReLU activations and has been trained", response)

    def test_handle_query_learning_rate(self):
        response = self.agent.handle_query("What is the learning rate?")
        self.assertIn("The agent's learning rate is:", response)

    def test_handle_query_batch_size(self):
        response = self.agent.handle_query("What is the batch size?")
        self.assertIn("The agent's batch size is:", response)

    def test_handle_query_episodes_trained(self):
        self.assertEqual(self.agent.episodes_trained, 10)  # Verify the state before querying
        response = self.agent.handle_query("How many episodes have been trained?")
        self.assertIn("The agent has been trained for", response)
        self.assertIn("episodes", response)
        response = self.agent.handle_query("What is the training progress?")
        self.assertIn("The agent has been trained for", response)
        self.assertIn("episodes", response)

    def test_handle_query_unknown(self):
        response = self.agent.handle_query("What is the meaning of life?")
        self.assertIn("I'm sorry, I don't understand the question.", response)

if __name__ == "__main__":
    unittest.main()
