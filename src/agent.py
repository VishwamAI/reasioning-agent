import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import spacy
import logging

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReasoningAgent:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state, _ = self.env.reset()
        self.done = False
        self.total_reward = 0
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.episodes_trained = 0
        self.context = {}  # Initialize context storage
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, done, truncated, info = result
            done = done or truncated
        self.total_reward += reward
        self.state = next_state
        self.done = done
        return next_state, reward, done, info

    def reset(self):
        self.state, _ = self.env.reset()
        self.done = False
        self.total_reward = 0

    def run_episode(self, max_steps=1000):
        self.reset()
        for _ in range(max_steps):
            if self.done:
                break
            action = self.choose_action(self.state)
            next_state, reward, done, _ = self.step(action)
            self.remember(self.state, action, reward, next_state, done)
            self.state = next_state
        return self.total_reward

    def train(self, episodes, batch_size):
        for e in range(episodes):
            try:
                total_reward = self.run_episode()
                if len(self.memory) > batch_size:
                    self.replay(batch_size)
                self.update_target_model()
                self.episodes_trained += 1
                logging.info(f"Episode {e+1}/{episodes}, Total reward: {total_reward}, Epsilon: {self.epsilon}")
            except Exception as e:
                logging.error(f"An error occurred during training: {e}")

    def save_model(self, filepath):
        self.model.save(filepath)
        logging.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        self.update_target_model()
        logging.info(f"Model loaded from {filepath}")

    def handle_query(self, query):
        doc = nlp(query)
        intents = [token.lemma_ for token in doc if token.pos_ in ["VERB", "NOUN", "ADJ"]]
        entities = [ent.text for ent in doc.ents]

        # Debug logging
        logging.info(f"Query: {query}")
        logging.debug(f"Intents: {intents}")
        logging.debug(f"Entities: {entities}")
        logging.debug(f"Done: {self.done}")
        logging.debug(f"Episodes Trained: {self.episodes_trained}")

        # Check for follow-up questions
        if "follow-up" in self.context:
            last_query = self.context["last_query"]
            last_response = self.context["last_response"]
            if any(intent in ["more", "detail", "explain"] for intent in intents):
                logging.debug(f"Returning: Providing more details about the last query: {last_query}")
                return f"Providing more details about the last query: {last_query}"

        # Consolidate intent checks
        if any(intent in ["done", "complete", "finished", "completion", "end", "over", "do", "finish"] for intent in intents):
            response = f"The episode is {'done' if self.done else 'not done'}."
        elif any(intent in ["learning", "rate", "speed"] for intent in intents):
            try:
                learning_rate = self.model.optimizer.learning_rate.numpy()
            except AttributeError:
                learning_rate = 0.001  # Default learning rate
            response = f"The agent's learning rate is: {learning_rate}."
        elif any(intent in ["training", "learning", "progress", "trained", "episodes trained", "training progress", "episodes", "train", "learning status", "status"] for intent in intents):
            response = f"The agent has been trained for {self.episodes_trained} episodes."
        elif any(intent in ["state"] for intent in intents):
            response = f"The current state is: {self.state}"
        elif any(intent in ["reward", "total"] for intent in intents):
            response = f"The total reward accumulated is: {self.total_reward}"
        elif any(intent in ["action", "move"] for intent in intents):
            action = self.choose_action(self.state)
            response = f"The chosen action is: {action}"
        elif any(intent in ["decision", "process"] for intent in intents):
            response = f"The agent's decision-making process involves selecting actions based on the highest Q-values predicted by the model."
        elif any(intent in ["hyperparameter", "parameter"] for intent in intents):
            response = f"The agent's hyperparameters are: gamma={self.gamma}, epsilon={self.epsilon}, epsilon_min={self.epsilon_min}, epsilon_decay={self.epsilon_decay}."
        elif any(intent in ["architecture", "structure", "model"] for intent in intents):
            if self.episodes_trained == 0:
                response = "The agent's model architecture consists of: Dense layers with ReLU activations."
            else:
                response = "The agent's model architecture consists of: Dense layers with ReLU activations and has been trained."
        elif any(intent in ["batch", "size"] for intent in intents):
            response = f"The agent's batch size is: {self.memory.maxlen}"
        elif any(intent in ["dataset", "data", "training data"] for intent in intents):
            response = "The agent is trained using the specified advancement dataset."
        elif entities:
            response = f"I'm sorry, I don't have information about: {', '.join(entities)}"
        else:
            # Check for greetings
            greetings = ["hi", "hello", "hey"]
            if any(greet in query.lower() for greet in greetings):
                response = "How can I assist you today?"
            else:
                response = "I'm sorry, I don't understand the question. Please ask about the state, reward, episode status, action, training status, hyperparameters, model architecture, learning rate, batch size, number of episodes trained, episode completion status, learning status, training progress, follow-up questions, or context maintenance."

        # Update context with the last query and response
        self.context["last_query"] = query
        self.context["last_response"] = response

        logging.info(f"Returning: {response}")
        return response

if __name__ == "__main__":
    agent = ReasoningAgent(env_name="CartPole-v1")
    agent.train(episodes=1000, batch_size=32)
    # Example queries
    logging.info(agent.handle_query("What is the current state?"))
    logging.info(agent.handle_query("What is the total reward?"))
    logging.info(agent.handle_query("Is the episode done?"))
    logging.info(agent.handle_query("What action should be taken?"))
