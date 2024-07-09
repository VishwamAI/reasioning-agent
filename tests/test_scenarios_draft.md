# Test Scenarios Draft for Reasoning Agent

## Follow-up Questions
1. **Query:** "What is the current state?"
   **Expected Response:** "The current state is: [state details]"
   **Follow-up Query:** "What action will you take?"
   **Expected Follow-up Response:** "The chosen action is: [action details]"

2. **Query:** "How many episodes have been trained?"
   **Expected Response:** "The agent has been trained for [number] episodes."
   **Follow-up Query:** "What is the training progress?"
   **Expected Follow-up Response:** "The agent has been trained for [number] episodes."

## Ambiguous Queries
1. **Query:** "Tell me about the training."
   **Expected Response:** "The agent has been trained for [number] episodes."

2. **Query:** "Is it done?"
   **Expected Response:** "The episode is [done/not done]."

## Complex Conversational Scenarios
1. **Query:** "If the episode is done, what was the total reward?"
   **Expected Response:** "The episode is [done/not done]. The total reward is: [reward details]"

2. **Query:** "What is the learning rate and batch size?"
   **Expected Response:** "The learning rate is [rate]. The batch size is [size]."

## Incorrect or Unclear Queries
1. **Query:** "What is the meaning of life?"
   **Expected Response:** "I'm sorry, I don't understand the question. Please ask about the state, reward, episode status, action, training status, hyperparameters, model architecture, learning rate, batch size, number of episodes trained, episode completion status, learning status, or training progress."

2. **Query:** "Can you help me?"
   **Expected Response:** "How can I assist you today?"

## Context Maintenance
1. **Query:** "What is the current state?"
   **Expected Response:** "The current state is: [state details]"
   **Follow-up Query:** "And what action will you take?"
   **Expected Follow-up Response:** "The chosen action is: [action details]"

2. **Query:** "How many episodes have been trained?"
   **Expected Response:** "The agent has been trained for [number] episodes."
   **Follow-up Query:** "And what is the learning rate?"
   **Expected Follow-up Response:** "The learning rate is [rate]."
