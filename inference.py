import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    
    forward_messages = [None] * len(observations)
    backward_messages = [None] * len(observations)
    marginals = [None] * len(observations)
    
    # Compute the forward messages
    forward_messages[0] = rover.Distribution({})
    initial_observation = observations[0]
    for hidden_state in all_possible_hidden_states:
        if initial_observation == None:
            initial_obs_prob = 1
        else:
            initial_obs_prob = observation_model(hidden_state)[initial_observation]
        prior_prob = prior_distribution[hidden_state]
        if (initial_obs_prob * prior_prob) != 0:
            forward_messages[0][hidden_state] = initial_obs_prob * prior_prob
    forward_messages[0].renormalize()
    
    for t in range(1, len(observations)):
        forward_messages[t] = rover.Distribution({})
        observation = observations[t]
        for current_hidden_state in all_possible_hidden_states:
            if observation == None:
                obs_prob = 1
            else:               
                obs_prob = observation_model(current_hidden_state)[observation]
            
            sum_prob = 0
            for prev_hidden_state in forward_messages[t-1]:
                sum_prob += forward_messages[t-1][prev_hidden_state] * transition_model(prev_hidden_state)[current_hidden_state]
            if (obs_prob * sum_prob) != 0:
                forward_messages[t][current_hidden_state] = obs_prob * sum_prob

        forward_messages[t].renormalize()
    
    # Compute the backward messages
    backward_messages[len(observations)-1] = rover.Distribution({})
    for hidden_state in all_possible_hidden_states:
        backward_messages[len(observations)-1][hidden_state] = 1
        
    for t in range(1, len(observations)):
        backward_messages[len(observations)-1-t] = rover.Distribution({})
        for current_hidden_state in all_possible_hidden_states:
            sum_prob = 0
            for next_hidden_state in backward_messages[len(observations)-1-t+1]:
                next_observation = observations[len(observations)-1-t+1]
                if next_observation == None:
                    next_obs_prob = 1
                else:
                    next_obs_prob = observation_model(next_hidden_state)[next_observation]
                sum_prob += backward_messages[len(observations)-1-t+1][next_hidden_state] * next_obs_prob * transition_model(current_hidden_state)[next_hidden_state]
            if sum_prob != 0:
                backward_messages[len(observations)-1-t][current_hidden_state] = sum_prob
        backward_messages[len(observations)-1-t].renormalize()
    
    # Compute the marginals
    for t in range(0, len(observations)): 
        marginals[t] = rover.Distribution({})    
        total_prob = 0
        for hidden_state in all_possible_hidden_states:
            if forward_messages[t][hidden_state] * backward_messages[t][hidden_state] != 0:
                marginals[t][hidden_state] = forward_messages[t][hidden_state] * backward_messages[t][hidden_state]
                total_prob += forward_messages[t][hidden_state] * backward_messages[t][hidden_state]
        for hidden_state in marginals[t].keys():
            marginals[t][hidden_state] /= total_prob

    return marginals




def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    viterbi_table = [{} for _ in range(len(observations))]
    backpointer_table = [{} for _ in range(len(observations))]

    # Initialization step
    for hidden_state in all_possible_hidden_states:
        if observations[0] is None:
            observation_prob = 1
        else:
            observation_prob = observation_model(hidden_state)[observations[0]]
        viterbi_table[0][hidden_state] = prior_distribution[hidden_state] * observation_prob
        backpointer_table[0][hidden_state] = None

    # Recursion step
    for t in range(1, len(observations)):
        for current_hidden_state in all_possible_hidden_states:
            max_prob = 0
            max_prev_state = None
            for prev_hidden_state in all_possible_hidden_states:
                prob = viterbi_table[t - 1][prev_hidden_state] * transition_model(prev_hidden_state)[current_hidden_state]
                if prob > max_prob:
                    max_prob = prob
                    max_prev_state = prev_hidden_state

            if observations[t] is None:
                observation_prob = 1
            else:
                observation_prob = observation_model(current_hidden_state)[observations[t]]

            viterbi_table[t][current_hidden_state] = max_prob * observation_prob
            backpointer_table[t][current_hidden_state] = max_prev_state

    # Termination step
    max_final_prob = max(viterbi_table[-1].values())
    best_final_state = None
    for hidden_state, prob in viterbi_table[-1].items():
        if prob == max_final_prob:
            best_final_state = hidden_state
            break

    # Traceback
    estimated_hidden_states = [None] * len(observations)
    estimated_hidden_states[-1] = best_final_state

    for t in range(len(observations) - 1, 0, -1):
        estimated_hidden_states[t - 1] = backpointer_table[t][estimated_hidden_states[t]]

    return estimated_hidden_states





if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = 30 #= num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(0, num_time_steps):
        print(estimated_states[time_step])

    
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
