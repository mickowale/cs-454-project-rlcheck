# AI Based software engineering term project
This project is privately copied from [rlcheck ](https://github.com/sameerreddy13/rlcheck) by Sameer Reddy et al.
Our project aims to make slight modifications to RL algorithm to achieve better performance.
The paper “Quickly Generating Diverse Valid Test Inputs with Reinforcement Learning “ addressed this gap by investigating a blackbox approach, called RLCheck, for guiding the generator to produce a large number of diverse valid inputs in a very short amount of time using reinforcement learning. However, this approach was found to have problems with early convergence, slightly biased output, and the inability to facilitate more complex outputs with many choice points and larger choice space. Hence, we aim to design a Deep Q-Network model for the guide that can handle more complex choice spaces and facilitate more exploration. We then evaluate DQN RLCheck against pure random input generation and the original RLCheck that uses a q-table.

