import configparser
import numpy as np

# Read configuration from file and store in dictionary
config_parser = configparser.ConfigParser()
config_parser.read('config.ini')

config = {
    'Prior_si': {0: float(config_parser['PRIORS']['si'].split(',')[0]), 1: float(config_parser['PRIORS']['si'].split(',')[1])},
    'Prior_rsi': {0: float(config_parser['PRIORS']['rsi'].split(',')[0]), 1: float(config_parser['PRIORS']['rsi'].split(',')[1])},
    'Prior_gan': {0: float(config_parser['PRIORS']['gan'].split(',')[0]), 1: float(config_parser['PRIORS']['gan'].split(',')[1])},
    'Prior_d': {0: float(config_parser['PRIORS']['d'].split(',')[0]), 1: float(config_parser['PRIORS']['d'].split(',')[1])},
    'Prior_sr': {0: float(config_parser['PRIORS']['sr'].split(',')[0]), 1: float(config_parser['PRIORS']['sr'].split(',')[1])},

    'CPT_mc': {(tuple(map(int, k.split(',')))): float(v) for k, v in config_parser['CPT_MC'].items()},
    'CPT_q': {(tuple(map(int, k.split(',')))): float(v) for k, v in config_parser['CPT_Q'].items()},
    'CPT_a': {(tuple(map(int, k.split(',')))): float(v) for k, v in config_parser['CPT_A'].items()},

    'tm_si': {(0, 0): float(config_parser['TRANSITION_MODELS']['tm_si'].split(',')[0]),
              (0, 1): float(config_parser['TRANSITION_MODELS']['tm_si'].split(',')[1]),
              (1, 0): float(config_parser['TRANSITION_MODELS']['tm_si'].split(',')[2]),
              (1, 1): float(config_parser['TRANSITION_MODELS']['tm_si'].split(',')[3])},

    'tm_rs': {(0, 0): float(config_parser['TRANSITION_MODELS']['tm_rs'].split(',')[0]),
              (0, 1): float(config_parser['TRANSITION_MODELS']['tm_rs'].split(',')[1]),
              (1, 0): float(config_parser['TRANSITION_MODELS']['tm_rs'].split(',')[2]),
              (1, 1): float(config_parser['TRANSITION_MODELS']['tm_rs'].split(',')[3])},

    'tm_gan': {(0, 0): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[0]),
               (0, 1): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[1]),
               (1, 0): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[2]),
               (1, 1): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[3])},

    'tm_sr': {(0, 0): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[0]),
               (0, 1): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[1]),
               (1, 0): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[2]),
               (1, 1): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[3])},
    'tm_d': {(0, 0): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[0]),
               (0, 1): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[1]),
               (1, 0): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[2]),
               (1, 1): float(config_parser['TRANSITION_MODELS']['tm_gan'].split(',')[3])},

    'N': int(config_parser['GENERAL']['N']),
    'T': int(config_parser['GENERAL']['T'])
}


###helper function
def priors(config):
    Prior_si=config.get("Prior_si")
    Prior_rsi=config.get("Prior_rsi")
    Prior_gan=config.get("Prior_gan")
    Prior_d=config.get("Prior_d")
    Prior_sr=config.get("Prior_sr")
    return Prior_si, Prior_rsi, Prior_gan,Prior_d,Prior_sr

def conditionals(config):
    CPT_mc=config.get('CPT_mc')
    CPT_q=config.get('CPT_q')
    CPT_a=config.get('CPT_a')
    return CPT_mc, CPT_q,CPT_a

def number_of_samples(config):
    N=config.get("N")
    return N

def timeslice(config):
    T=config.get("T")
    return T

def transition_model(config):
    tm_si=config.get("tm_si")
    tm_rs=config.get("tm_rs")
    tm_gan = config.get("tm_gan")
    tm_sr = config.get("tm_sr")
    tm_d = config.get("tm_d")
    return tm_si,tm_rs,tm_gan, tm_sr,tm_d

####Particle filter algorithm function
def initialize(N:int,Prior_si,Prior_rsi,Prior_gan,T=0):
    # Sample N values from each prior distribution
    np.random.seed(45)
    sampled_si = np.random.choice(list(Prior_si.keys()), N, p=list(Prior_si.values()))
    sampled_rsi = np.random.choice(list(Prior_rsi.keys()), N, p=list(Prior_rsi.values()))
    sampled_gan = np.random.choice(list(Prior_gan.keys()), N, p=list(Prior_gan.values()))

    # Get the corresponding probabilities
    sampled_prob_si = [Prior_si[val] for val in sampled_si]
    sampled_prob_rsi = [Prior_rsi[val] for val in sampled_rsi]
    sampled_prob_gan = [Prior_gan[val] for val in sampled_gan]

    # Store the results in a dictionary
    sampled_values = {
        'si': sampled_si,
        'rsi': sampled_rsi,
        'gan': sampled_gan
    }

    sampled_probabilities = {
        'si': sampled_prob_si,
        'rsi': sampled_prob_rsi,
        'gan': sampled_prob_gan
    }
    sampled_weights=[1/N]*N
    return sampled_values, sampled_probabilities, sampled_weights

def propagate():
    pass




def updated_particles(sampled_values):
    pass

def mc_sequence(sampled_values_t, sampled_probabilities_t, sampled_weights):
    # Extract particle values
    si_values = sampled_values_t['si']
    rsi_values = sampled_values_t['rsi']
    gan_values = sampled_values_t['gan']

    # Determine sampled values and probabilities
    sampled_values = list(zip(si_values, rsi_values, gan_values))
    sampled_probabilities = np.array([CPT_mc[tuple(val)] for val in sampled_values])

    # Normalize probabilities to obtain weights
    sampled_weights *= sampled_probabilities / np.sum(sampled_probabilities)

    # Randomly pick values of mc as 0 or 1
    mc_values = np.random.choice([0, 1], size=len(sampled_values))

    # Adjust probabilities for mc = 0 or 1
    adjusted_probabilities = np.where(mc_values == 0, 1 - sampled_probabilities, sampled_probabilities)

    # Append mc values to sampled values
    sampled_values_with_mc = [(*val, mc) for val, mc in zip(sampled_values, mc_values)]

    return sampled_values_with_mc, adjusted_probabilities, sampled_weights

def q_sequence(sampled_values, sampled_probabilities, sampled_weights, Prior_d, CPT_q):
    # Sample D values based on Prior_d probabilities
    mc_values = [tup[-1] for tup in sampled_values]  # Extracting MC values

    d_values = np.random.choice([0, 1], size=len(mc_values), p=[Prior_d[0], Prior_d[1]])

    # Sample Q values based on P(Q|MC) table
    q_values = [np.random.choice([0, 1], p=[1 - CPT_q[(mc, d)], CPT_q[(mc, d)]]) for mc, d in zip(mc_values, d_values)]

    # Append D and Q values to sampled values
    sampled_values_with_d_q = [(*val, d, q) for val, d, q in zip(sampled_values, d_values, q_values)]

    # Update probabilities based on Q values
    updated_probabilities = np.array(
        [CPT_q[(mc, d)] if q == 1 else (1 - CPT_q[(mc, d)]) for mc, d, q in zip(mc_values, d_values, q_values)])

    # Normalize probabilities to obtain updated weights
    sampled_weights *= updated_probabilities / np.sum(updated_probabilities)

    return sampled_values_with_d_q, updated_probabilities, sampled_weights

def a_sequence(sampled_values, sampled_probabilities, sampled_weights, Prior_sr, CPT_a):
    # Sample SR values based on Prior_sr probabilities
    q_values = [tup[-1] for tup in sampled_values]  # Extracting Q values

    sr_values = np.random.choice([0, 1], size=len(q_values), p=[Prior_d[0], Prior_d[1]])

    # Sample Q values based on P(Q|MC) table
    q_values = [np.random.choice([0, 1], p=[1 - CPT_a[(mc, d)], CPT_a[(mc, d)]]) for mc, d in zip(q_values, sr_values)]

    # Append D and Q values to sampled values
    sampled_values_with_sr_a = [(*val, d, q) for val, d, q in zip(sampled_values, sr_values, q_values)]

    # Update probabilities based on Q values
    updated_probabilities = np.array(
        [CPT_a[(mc, d)] if q == 1 else (1 - CPT_a[(mc, d)]) for mc, d, q in zip(q_values, sr_values, q_values)])

    # Normalize probabilities to obtain updated weights
    sampled_weights *= updated_probabilities / np.sum(updated_probabilities)

    return sampled_values_with_sr_a, updated_probabilities, sampled_weights

def resample(sampled_values, sampled_weights, N):
    sampled_weights = sampled_weights / np.sum(sampled_weights)
    # Perform resampling using weighted choice
    indices = np.random.choice(len(sampled_values), size=N, p=sampled_weights, replace=True)
    resampled_particles = [sampled_values[i] for i in indices]
    return resampled_particles

def importance_resample(sampled_values, sampled_weights, N):
    # Normalize the weights to sum to 1
    normalized_weights = sampled_weights / np.sum(sampled_weights)
    # Perform resampling using the normalized weights
    indices = np.random.choice(len(sampled_values), size=N, p=normalized_weights, replace=True)
    # Resample particles based on the selected indices
    resampled_particles = [sampled_values[i] for i in indices]
    # Optionally, you can reset or adjust the weights of the resampled particles, typically all weights are reset
    resampled_weights = np.ones(N) / N  # Equal weight for each resampled particle
    return resampled_particles, resampled_weights

def next_initialization(tm_si,tm_rs,tm_gan,N,sampled_values,sampled_weights):
    sampled_values = []
    sampled_probabilities = []

    states = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for _ in range(N):
        state_si = states[np.random.randint(len(states))]
        state_rs = states[np.random.randint(len(states))]
        state_gan = states[np.random.choice(len(states))]

        val_si, prob_si = state_si[1], tm_si[state_si]
        val_rs, prob_rs = state_rs[1], tm_rs[state_rs]
        val_gan, prob_gan = state_gan[1], tm_gan[state_gan]

        sampled_values.append((val_si, val_rs, val_gan))
        sampled_probabilities.append(prob_si * prob_rs * prob_gan)

    return sampled_values, sampled_probabilities, sampled_weights

    return new_sampled_values, sampled_probabilities, sampled_weights

def mc_sequence_next(sampled_values, sampled_probabilities, sampled_weights):
    sampled_probabilities = np.array([CPT_mc[tuple(val)] for val in sampled_values])

    # Normalize probabilities to obtain weights
    sampled_weights *= sampled_probabilities / np.sum(sampled_probabilities)

    # Randomly pick values of mc as 0 or 1
    mc_values = np.random.choice([0, 1], size=len(sampled_values))

    # Adjust probabilities for mc = 0 or 1
    adjusted_probabilities = np.where(mc_values == 0, 1 - sampled_probabilities, sampled_probabilities)

    # Append mc values to sampled values
    sampled_values_with_mc = [(*val, mc) for val, mc in zip(sampled_values, mc_values)]

    return sampled_values_with_mc, adjusted_probabilities, sampled_weights

def q_sequence_next(sampled_values, sampled_probabilities, sampled_weights, tm_d, CPT_q):
    # Extract MC values from sampled_values (assuming MC is the last element)
    mc_values = [tup[-1] for tup in sampled_values]

    # Correctly sample D based on conditional probabilities given MC
    d_values = np.array([
        np.random.choice([0, 1], p=[tm_d[(mc, 0)], tm_d[(mc, 1)]])
        for mc in mc_values
    ])

    # Sample Q values based on CPT_q conditioned on (MC, D)
    q_values = np.array([
        np.random.choice([0, 1], p=[1 - CPT_q[(mc, d)], CPT_q[(mc, d)]])
        for mc, d in zip(mc_values, d_values)
    ])

    # Append D and Q to sampled_values
    sampled_values_with_d_q = [
        (*val, d, q) for val, d, q in zip(sampled_values, d_values, q_values)
    ]

    # Update probabilities based on conditional sampling
    updated_probabilities = np.array([
        tm_d[(mc, d)] * (CPT_q[(mc, d)] if q == 1 else (1 - CPT_q[(mc, d)]))
        for mc, d, q in zip(mc_values, d_values, q_values)
    ])

    # Normalize updated probabilities to update sampled_weights
    sampled_weights *= updated_probabilities
    sampled_weights /= np.sum(sampled_weights)

    return sampled_values_with_d_q, updated_probabilities, sampled_weights


def a_sequence_next(sampled_values, sampled_probabilities, sampled_weights, tm_sr, CPT_a):
    # Extract MC values from sampled_values (assuming MC is the last element in each tuple)
    mc_values = [tup[-1] for tup in sampled_values]

    # Sample SR based on Prior_sr conditioned on MC
    sr_values = np.array([
        np.random.choice([0, 1], p=[tm_sr[(mc, 0)], tm_sr[(mc, 1)]]) for mc in mc_values
    ])

    # Sample Q based on CPT_a conditioned on MC and SR
    q_values = np.array([
        np.random.choice([0, 1], p=[1 - CPT_a[(mc, sr)], CPT_a[(mc, sr)]])
        for mc, sr in zip(mc_values, sr_values)
    ])

    # Append SR and Q values to sampled values
    sampled_values_with_sr_a = [
        (*val, sr, q) for val, sr, q in zip(sampled_values, sr_values, q_values)
    ]

    # Update probabilities based on sampled SR and Q values
    updated_probabilities = np.array([
        (tm_sr[(mc, sr)]) * (CPT_a[(mc, sr)] if q == 1 else (1 - CPT_a[(mc, sr)]))
        for mc, sr, q in zip(mc_values, sr_values, q_values)
    ])

    # Normalize updated probabilities to obtain new weights
    sampled_weights = sampled_weights * updated_probabilities
    sampled_weights /= np.sum(sampled_weights)

    return sampled_values_with_sr_a, updated_probabilities, sampled_weights



##queries
def compute_conditional_probability(updated_samples, target_var:str, target_time:int, condition_var:str, condition_time:int, condition_value:int):
    var_indices = {'si': 0, 'rsi': 1, 'mc': 2, 'gan': 3, 'd': 4, 'q': 5, 'sr': 6, 'a': 7}

    particles_condition_time = updated_samples[condition_time]
    particles_target_time = updated_samples[target_time]

    indices_condition_met = [idx for idx, particle in enumerate(particles_condition_time) if particle[var_indices[condition_var]] == condition_value]

    count_target_var_1_given_condition = sum(particles_target_time[idx][var_indices[target_var]] for idx in indices_condition_met)
    total_condition_met = len(indices_condition_met)

    probability = count_target_var_1_given_condition / total_condition_met if total_condition_met > 0 else None

    return probability

def compute_conditional_probability_mult(updated_samples, target_var, target_time, condition_vars, condition_time, condition_values):
    var_indices = {'si': 0, 'rs': 1, 'gan': 2, 'mc': 3, 'd': 4, 'q': 5, 'sr': 6, 'a': 7}

    condition_particles = updated_samples[condition_time]
    target_particles = updated_samples[target_time]

    matching_indices = [idx for idx, particle in enumerate(condition_particles)
                        if all(particle[var_indices[var]] == val for var, val in zip(condition_vars, condition_values))]

    if not matching_indices:
        return None

    probability = sum(target_particles[idx][var_indices[target_var]] for idx in matching_indices) / len(matching_indices)

    return probability
'''
def compute_do_probability(updated_samples, target_var, target_time, intervention_var, intervention_time, intervention_value):
    var_indices = {'si': 0, 'rs': 1, 'gan': 2, 'mc': 3, 'd': 4, 'q': 5, 'sr': 6, 'a': 7}

    pre_intervention_particles = updated_samples[intervention_time - 1]
    post_intervention_particles = updated_samples[target_time]

    matching_indices = [idx for idx, particle in enumerate(updated_samples[intervention_time]) if particle[var_indices[intervention_var]] == intervention_value]

    if not matching_indices:
        return None

    probability = sum(updated_samples[target_time][idx][var_indices[target_var]] for idx in matching_indices) / len(matching_indices)

    return probability
'''

'''ignore'''
def compute_do_probability_test1(updated_samples, target_var, target_time, intervention_var, intervention_time, intervention_value):
    var_indices = {'si': 0, 'rs': 1, 'gan': 2, 'mc': 3, 'q': 4, 'd': 5, 'sr': 6, 'a': 7}

    intervention_particles = updated_samples[intervention_time]
    target_particles = updated_samples[target_time]

    matching_indices = [idx for idx, particle in enumerate(intervention_particles) if particle[var_indices[intervention_var]] == intervention_value]

    if not matching_indices:
        return None

    probability = sum(target_particles[idx][var_indices[target_var]] for idx in matching_indices) / len(matching_indices)

    return probability

'''
parameters:
updated_samples: Samples after PFA is applied
target_var: hypothesis
target_time: timeslice to measure the hypothesis
intervention_var: condition or given
intervention_time: timeslice for condition
intervention_value: is the intervention, low or high- 0/1
marginalize_var: Variable on which backdoor criterion is applied
marginalize_time: timeslice in which the variable is being marginalised

returns the estimand P(X|do(y))
'''
def compute_do_probability(updated_samples, target_var, target_time, intervention_var, intervention_time, intervention_value, marginalize_var, marginalize_time):
    var_indices = {'si': 0, 'rs': 1, 'gan': 2, 'mc': 3, 'q': 4, 'd': 5, 'sr': 6, 'a': 7}

    particles_marginalize_time = updated_samples[marginalize_time]
    particles_intervention_time = updated_samples[intervention_time]
    particles_target_time = updated_samples[target_time]

    marginalize_idx = var_indices[marginalize_var]
    intervention_idx = var_indices[intervention_var]
    target_idx = var_indices[target_var]

    marginalize_values = set(p[marginalize_idx] for p in particles_marginalize_time)

    probability = 0.0

    for marginalize_value in marginalize_values:
        indices_marginalize = [idx for idx, particle in enumerate(particles_marginalize_time)
                               if particle[marginalize_idx] == marginalize_value]

        if not indices_marginalize:
            continue

        prob_marginalize = len(indices_marginalize) / len(particles_marginalize_time)

        indices_intervention_given_marginalize = [idx for idx in indices_marginalize
                                                  if particles_intervention_time[idx][intervention_idx] == intervention_value]

        if not indices_intervention_given_marginalize:
            continue

        prob_target_given_conditions = sum(particles_target_time[idx][target_idx] for idx in indices_intervention_given_marginalize) / len(indices_intervention_given_marginalize)

        probability += prob_target_given_conditions * prob_marginalize

    return probability

'''
For multiple do operations
'''
def compute_do_probability_mult(updated_samples, target_var, target_time, intervention_var, intervention_time, intervention_value, marginalized_var, marginalized_time):
    var_indices = {'si': 0, 'rs': 1, 'gan': 2, 'mc': 3, 'q': 4, 'd': 5, 'sr': 6, 'a': 7}

    particles_marginalize_time = updated_samples[marginalized_time]
    particles_intervention_time = updated_samples[intervention_time]
    particles_target_time = updated_samples[target_time]

    marginalize_idx = var_indices[marginalized_var]
    intervention_idx = var_indices[intervention_var]
    target_idx = var_indices[target_var]

    marginalize_values = set(particle[marginalize_idx] for particle in particles_marginalize_time)

    probability = 0.0

    for marginalize_value in marginalize_values:
        indices_marginalize = [idx for idx, particle in enumerate(particles_marginalize_time)
                               if particle[marginalize_idx] == marginalize_value]

        prob_marginalize = len(indices_marginalize) / len(particles_marginalize_time)

        indices_intervention = [idx for idx in indices_marginalize
                                if particles_intervention_time[idx][var_indices[intervention_var]] == intervention_value]

        if not indices_intervention:
            continue

        prob_target_given_conditions = sum(particles_target_time[idx][target_idx] for idx in indices_intervention) / len(indices_intervention)

        probability += prob_target_given_conditions * prob_marginalize

    return probability


####implementation of particle filter algorithm
updated_particle_log=[] #save all particles at the end of resampling at every timeslice
update_particles_dict={}
Prior_si, Prior_rsi, Prior_gan,Prior_d,Prior_sr= priors(config)
CPT_mc, CPT_q,CPT_a=conditionals(config)
N=number_of_samples(config)
T=timeslice(config)
t=0
tm_si,tm_rs,tm_gan, tm_sr,tm_d=transition_model(config)
sampled_values, sampled_probabilities, sampled_weights=initialize(N,Prior_si,Prior_rsi,Prior_gan,0) #for timeslice 0
#print(sampled_values,sampled_probabilities,sampled_weights)
#print(sampled_values)
#print(transition_lookup_mc(sampled_values,sampled_probabilities,sampled_weights))
sampled_values,sampled_probabilities,sampled_weights=mc_sequence(sampled_values, sampled_probabilities, sampled_weights)
#print(sampled_values)
sampled_values,sampled_probabilities,sampled_weights=q_sequence(sampled_values, sampled_probabilities, sampled_weights, Prior_d, CPT_q)
#print(sampled_values)
sampled_values,sampled_probabilities,sampled_weights=a_sequence(sampled_values, sampled_probabilities, sampled_weights, Prior_sr, CPT_a)
#print(sampled_values)
#print(sampled_weights)

resampled_particles, resampled_weights=importance_resample(sampled_values, sampled_weights, N)
updated_particle_log.append(resampled_particles)
#my_dict['new_key'] = 'new_value'
update_particles_dict[t]=resampled_particles

for t in range(1, T):
    sampled_values,sampled_weights,sampled_probabilities=next_initialization(tm_si,tm_rs,tm_gan,N,resampled_particles,resampled_weights)
    #print(sampled_values)
    sampled_values, sampled_probabilities, sampled_weights = mc_sequence_next(sampled_values, sampled_probabilities,
                                                                         sampled_weights)
    #print(sampled_values)
    sampled_values, sampled_probabilities, sampled_weights = q_sequence_next(sampled_values, sampled_probabilities,
                                                                        sampled_weights, tm_d, CPT_q)
    #print(sampled_values)
    sampled_values, sampled_probabilities, sampled_weights = a_sequence_next(sampled_values, sampled_probabilities,
                                                                        sampled_weights, tm_sr, CPT_a)
    resampled_particles, resampled_weights = importance_resample(sampled_values, sampled_weights, N)
    updated_particle_log.append(resampled_particles)
    update_particles_dict[t]=resampled_particles
print("~~~~~~~~~~~")
#print(update_particles_dict)

###########query analysis
print(compute_conditional_probability(update_particles_dict,"q",3,"d",2,0)) #updated_samples, target_var:str, target_time:int, condition_var:str, condition_time:int, condition_value:int

###########causal query analysis
#print(compute_do_probability(update_particles_dict,"a",3,"si",2,0)) # compute_do_probability(updated_samples, target_var, target_time, intervention_var, intervention_time, intervention_value)#updated_samples, target_var='a', target_time=2,intervention_var='si', intervention_time=1, intervention_value=0
print(compute_do_probability(update_particles_dict,"a",3,"si",2,0,"si",0))




