import unittest
from src.agent import ReasoningAgent

class TestReasoningAgentConversation(unittest.TestCase):

    def setUp(self):
        self.agent = ReasoningAgent()

    def test_greeting(self):
        query = "hi"
        response = self.agent.handle_query(query)
        self.assertEqual(response, "How can I assist you today?")

    def test_episode_status(self):
        query = "Is the episode done?"
        response = self.agent.handle_query(query)
        self.assertIn("The episode is", response)

    def test_learning_rate(self):
        query = "What is the learning rate?"
        response = self.agent.handle_query(query)
        self.assertIn("The agent's learning rate is:", response)

    def test_training_progress(self):
        query = "How many episodes have you trained?"
        response = self.agent.handle_query(query)
        self.assertIn("The agent has been trained for", response)

    def test_current_state(self):
        query = "What is the current state?"
        response = self.agent.handle_query(query)
        self.assertIn("The current state is:", response)

    def test_total_reward(self):
        query = "What is the total reward?"
        response = self.agent.handle_query(query)
        self.assertIn("The total reward accumulated is:", response)

    def test_chosen_action(self):
        query = "What action will you take?"
        response = self.agent.handle_query(query)
        self.assertIn("The chosen action is:", response)

    def test_decision_process(self):
        query = "How do you make decisions?"
        response = self.agent.handle_query(query)
        self.assertIn("The agent's decision-making process involves", response)

    def test_hyperparameters(self):
        query = "What are your hyperparameters?"
        response = self.agent.handle_query(query)
        self.assertIn("The agent's hyperparameters are:", response)

    def test_unrecognized_query(self):
        query = "Tell me a joke"
        response = self.agent.handle_query(query)
        self.assertIn("I'm sorry, I don't understand the question.", response)

if __name__ == '__main__':
    unittest.main()
