import torch

BUFFER_SIZE = int(1e5)     # replay buffer size
BATCH_SIZE = 512           # minibatch size
GAMMA = 0.99               # discount factor
TAU = 1e-3                 # for soft update of target parameters
LR_ACTOR = 1e-3            # learning rate of the actor 
LR_CRITIC = 1e-3           # learning rate of the critic
WEIGHT_DECAY = 0           # L2 weight decay
UPDATE_EVERY = 2           # Time steps between learning from experiences
NUM_UPDATE = 4             # Number of updates performed every UPDATE_EVERY
MAX_EPISODES = 2000        # Maximum number of episodes
EPSILON = 0.1              # Attenuation factor
EPSILON_START = 0.1        # Initial value of attenuation added to noise
EPSILON_DECAY_RATE = 0     # Attenuation factor for noise #1e-3
RANDOM_SEED = 0            # Random seed
    
OU_MU = 0                  # Noise Parameter
OU_SIGMA = 0.2             # Noise Parameter
OU_THETA = 0.15            # Noise Parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class constParams():
    """Saving the hyperparameters in a class of their own."""
    
    def __init__(self):
        """Class initialization."""
        
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.lr_actor = LR_ACTOR
        self.lr_critic = LR_CRITIC
        self.weight_decay = WEIGHT_DECAY
        self.update_every = UPDATE_EVERY
        self.num_update = NUM_UPDATE
        self.max_episodes = MAX_EPISODES
        self.epsilon_start = EPSILON_START
        self.epsilon_decay_rate = EPSILON_DECAY_RATE
        self.random_seed = RANDOM_SEED
        
        self.ou_mu = OU_MU
        self.ou_sigma = OU_SIGMA
        self.ou_theta = OU_THETA
        
        
        
