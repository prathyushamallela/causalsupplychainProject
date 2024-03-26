import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, transition_model, observation_model, initial_state):
        self.num_particles = num_particles
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.particles = [initial_state.copy() for _ in range(num_particles)]
        self.weights = np.ones(num_particles) / num_particles

    def predict(self):
        for i in range(self.num_particles):
            self.particles[i] = self.transition_model(self.particles[i])

    def update(self, observation):
        likelihoods = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            likelihoods[i] = self.observation_model(observation, self.particles[i])
        self.weights *= likelihoods
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, replace=True, p=self.weights)
        self.particles = [self.particles[i] for i in indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def particle_filter(self, observations):
        for observation in observations:
            self.predict()
            self.update(observation)
            self.resample()
            yield np.mean(self.particles, axis=0)

# Example usage:

# Define transition and observation models
def transition_model(state):
    # Simple transition model: state transition is independent of previous state
    return state

def observation_model(observation, state):
    # Simple observation model: assume a noisy observation
    if state == 0:
        return 0.8 if observation == 'A' else 0.2
    else:
        return 0.2 if observation == 'A' else 0.8

# Define initial state
initial_state = np.random.choice([0, 1], p=[0.5, 0.5])

# Define observations
observations = ['A', 'B', 'A', 'A', 'B']

# Create Particle Filter
num_particles = 1000
particle_filter = ParticleFilter(num_particles, transition_model, observation_model, initial_state)

# Run particle filter
estimated_states = list(particle_filter.particle_filter(observations))

print("Estimated States:", estimated_states)