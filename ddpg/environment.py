from unityagents import UnityEnvironment

class TennisEnvironment:
    
    """An adapter for an UnityEnvironment object to have an interface similar to the one of OpenGym"""
    
    def __init__(self, env, train_mode=True):
        """
        Create a Tennis environment using a given UnityEnvironment
        """
        self.env = env
        self.brain_name = env.brain_names[0]
        brain = env.brains[self.brain_name]
        env_info = env.reset(train_mode)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.action_size = brain.vector_action_space_size
        self.state_size  = env_info.vector_observations.shape[1]
        
    def step(self, actions):
        """
        
        """
        env_info   = self.env.step(actions)[self.brain_name]    # sends the action of each agent for the environment
        states = env_info.vector_observations                   # get the next state for each agent
        rewards =  env_info.rewards                             # get the reward for each agent
        dones = env_info.local_done                             # check if any episode has finished
        return states, rewards, dones
    
    def reset(self, train_mode=True):
        """
          Reset the environment and returns the new states
        """
        env_info = self.env.reset(train_mode)[self.brain_name]
        states = env_info.vector_observations
        return states
    
    def close(self):
        self.env.close()