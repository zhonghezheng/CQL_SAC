import os
import pandas as pd
import pickle
import numpy as np

if __name__ == '__main__':
    filename = os.path.join(os.path.dirname(__file__), 'data_5000.csv')
    data_raw = pd.read_csv(filename)
    data_processed = dict()
    # converting from obj to list
    # for i in data['actions']:
    #     print(type(i))
    #episode,actions,states,previous_actions,previous_states
    data_processed['states'] = [np.array(eval(i)).astype(np.float32) for i in data_raw['previous_states']]
    data_processed['next_states'] = [np.array(eval(i)).astype(np.float32) for i in data_raw['states']]
    data_processed['actions'] = [np.array([i]).astype(np.float32) for i in data_raw['previous_actions']]
    data_processed['next_actions'] = [np.array([i]).astype(np.float32) for i in data_raw['actions']]

    # populating done signals 
    done = [np.array([0.0]).astype(np.float32) if data_raw['episode'][i] == data_raw['episode'][i+1] else np.array([1.0]).astype(np.float32) for i in range(len(data_raw)-1)]
    done.append(np.array([1.0]).astype(np.float32))
    data_processed['done'] = done

    # calculating reward based on state
    rewards = []
    remove_state = False
    i = 0
    while i < len(data_processed['states']):
    # for state, action, d in zip(data_processed['states'], data_processed['actions'], data_processed['done']):
        if remove_state:
            data_processed['states'].pop(i)
            data_processed['next_states'].pop(i)
            data_processed['actions'].pop(i)
            data_processed['next_actions'].pop(i)
            data_processed['done'].pop(i)
            remove_state = False
            # print()
        else:
            action = data_processed['actions'][i]
            state = data_processed['states'][i]
            theta = state[0]
            theta_dot = state[1]
            rewards.append(np.float32(-(theta*theta + 0.1*theta_dot*theta_dot + 0.001 * action[0] * action[0])))
            # print(-(theta*theta + 0.1*theta_dot*theta_dot + 0.001 * action[0] * action[0]))

        
            if data_processed['done'][i] == 1: 
                remove_state = True
            i += 1

    data_processed['rewards'] = rewards
   

    # actions,state,next_state,reward,done
    fn = os.path.join(os.path.dirname(__file__), 'data_augmented.txt')
    with open(fn, 'wb') as f:
        pickle.dump(data_processed, f)
