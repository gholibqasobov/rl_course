import json
import numpy as np

# Load MDP data
with open('mdp_data.json', 'r') as f:
    mdp_data = json.load(f)

states = mdp_data['states']
actions = mdp_data['actions']
transitions = mdp_data['transitions']
gamma = mdp_data['gamma']

# Initialize random policy
policy = {s: np.random.choice(actions) for s in states}

def policy_evaluation(policy, theta=1e-6):
    """Evaluate a given policy."""
    V = {s: 0 for s in states}
    while True:
        delta = 0
        for s in states:
            v = V[s]
            a = policy[s]
            V[s] = sum(
                t['prob'] * (t['reward'] + gamma * V[t['next_state']])
                for t in transitions[str(s)][a]
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement(V):
    """Improve policy using value function."""
    policy_stable = True
    new_policy = policy.copy()
    
    for s in states:
        old_action = policy[s]
        # Compute action values
        action_values = {}
        for a in actions:
            action_values[a] = sum(
                t['prob'] * (t['reward'] + gamma * V[t['next_state']])
                for t in transitions[str(s)][a]
            )
        # Choose best action
        best_action = max(action_values, key=action_values.get)
        new_policy[s] = best_action
        
        if old_action != best_action:
            policy_stable = False
    
    return new_policy, policy_stable

def policy_iteration():
    global policy
    while True:
        V = policy_evaluation(policy)
        policy, stable = policy_improvement(V)
        if stable:
            return policy, V

# Run Policy Iteration
optimal_policy, optimal_value = policy_iteration()

# Output
print("Optimal Policy:")
for s in states:
    print(f"State {s}: {optimal_policy[s]}")

print("\nOptimal Value Function:")
for s in states:
    print(f"State {s}: {optimal_value[s]:.4f}")
