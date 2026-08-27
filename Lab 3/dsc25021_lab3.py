# ============================================================
# REINFORCEMENT LEARNING LAB - GRIDWORLD
# Policy Evaluation and Value Iteration
# ============================================================

import random


# ============================================================
# 1. GRIDWORLD ENVIRONMENT
# ============================================================

states = ["S1", "S2", "S3", "S4", "S5", "G"]

# Grid layout
#
#       Column 1   Column 2   Column 3
#
#       G          S1         S2
#       S5         S4         S3
#

grid = [
    ["G", "S1", "S2"],
    ["S5", "S4", "S3"]
]

# Coordinates of each state
positions = {
    "G":  (0, 0),
    "S1": (0, 1),
    "S2": (0, 2),
    "S5": (1, 0),
    "S4": (1, 1),
    "S3": (1, 2)
}

# Available actions
actions = ["U", "D", "L", "R"]

# Discount factor
gamma = 0.8

# Rewards
STEP_REWARD = -2
GOAL_REWARD = 12

# Starting state
START_STATE = "S3"

# Convergence threshold
THETA = 0.001


# ============================================================
# 2. ENVIRONMENT TRANSITION FUNCTION
# ============================================================

def step(state, action):

    # Goal is terminal
    if state == "G":
        return "G", 0, True

    # Movement corresponding to each action
    movement = {
        "U": (-1, 0),
        "D": (1, 0),
        "L": (0, -1),
        "R": (0, 1)
    }

    row, column = positions[state]

    dr, dc = movement[action]

    new_position = (
        row + dr,
        column + dc
    )

    # Default:
    # If movement is invalid, stay in the same state
    next_state = state

    # Find the state at the new position
    for name, coordinates in positions.items():

        if coordinates == new_position:
            next_state = name
            break

    # If goal is reached
    if next_state == "G":
        return next_state, GOAL_REWARD, True

    # Normal movement
    return next_state, STEP_REWARD, False


# ============================================================
# 3. DISPLAY GRIDWORLD
# ============================================================

print("=" * 65)
print("GRIDWORLD ENVIRONMENT")
print("=" * 65)

for row in grid:

    for state in row:

        if state == "G":
            print("[  G  ]", end=" ")

        else:
            print(f"[ {state} ]", end=" ")

    print()

print()
print("Starting State :", START_STATE)
print("Goal State     : G")
print("Step Reward    :", STEP_REWARD)
print("Goal Reward    :", GOAL_REWARD)
print("Discount Factor:", gamma)


# ============================================================
# 4. GIVEN POLICY FOR POLICY EVALUATION
# ============================================================

# Fixed policy for Task 2
#
# S1 -> Left
# S2 -> Right
# S3 -> Up
# S4 -> Right
# S5 -> Up

given_policy = {
    "S1": "L",
    "S2": "R",
    "S3": "U",
    "S4": "R",
    "S5": "U"
}

print("\n")
print("=" * 65)
print("GIVEN POLICY")
print("=" * 65)

for state in states:

    if state == "G":
        print("G  -> Terminal")

    else:
        print(
            f"{state} -> {given_policy[state]}"
        )


# ============================================================
# 5. POLICY EVALUATION
# ============================================================

print("\n")
print("=" * 65)
print("TASK 2 - POLICY EVALUATION")
print("=" * 65)

# Initial value function
V = {
    state: 0.0
    for state in states
}

# Store observations
policy_evaluation_history = []

# Perform first five iterations
for iteration in range(1, 6):

    new_V = V.copy()

    delta = 0.0

    for state in states:

        # Terminal state
        if state == "G":

            new_V[state] = 0.0

            continue

        # Action selected by the fixed policy
        action = given_policy[state]

        # Perform transition
        next_state, reward, done = step(
            state,
            action
        )

        # Bellman expectation update
        if done:

            new_value = reward

        else:

            new_value = (
                reward +
                gamma * V[next_state]
            )

        new_V[state] = new_value

        # Maximum change
        delta = max(
            delta,
            abs(new_value - V[state])
        )

    # Update value function
    V = new_V

    # Convergence status
    if delta < THETA:
        status = "Converged"

    else:
        status = "Not Converged"

    policy_evaluation_history.append(
        (
            iteration,
            delta,
            status,
            V.copy()
        )
    )


# ============================================================
# 6. PRINT POLICY EVALUATION TABLE
# ============================================================

print()

print(
    f"{'Iteration':<12}"
    f"{'Maximum Change':<18}"
    f"{'Status':<18}"
)

print("-" * 48)

for iteration, delta, status, values in policy_evaluation_history:

    print(
        f"{iteration:<12}"
        f"{delta:<18.4f}"
        f"{status:<18}"
    )


# ============================================================
# 7. PRINT VALUE FUNCTION AT EACH ITERATION
# ============================================================

print("\n")
print("VALUE FUNCTION DURING POLICY EVALUATION")
print("-" * 65)

print(
    f"{'Iteration':<12}"
    f"{'S1':<12}"
    f"{'S2':<12}"
    f"{'S3':<12}"
    f"{'S4':<12}"
    f"{'S5':<12}"
)

print("-" * 65)

for iteration, delta, status, values in policy_evaluation_history:

    print(
        f"{iteration:<12}"
        f"{values['S1']:<12.3f}"
        f"{values['S2']:<12.3f}"
        f"{values['S3']:<12.3f}"
        f"{values['S4']:<12.3f}"
        f"{values['S5']:<12.3f}"
    )


# ============================================================
# 8. POLICY EVALUATION UNTIL CONVERGENCE
# ============================================================

V_eval = {
    state: 0.0
    for state in states
}

convergence_iteration = 0

while True:

    convergence_iteration += 1

    new_V = V_eval.copy()

    delta = 0.0

    for state in states:

        if state == "G":

            new_V[state] = 0.0

            continue

        action = given_policy[state]

        next_state, reward, done = step(
            state,
            action
        )

        if done:

            new_value = reward

        else:

            new_value = (
                reward +
                gamma * V_eval[next_state]
            )

        new_V[state] = new_value

        delta = max(
            delta,
            abs(new_value - V_eval[state])
        )

    V_eval = new_V

    if delta < THETA:

        break


# ============================================================
# 9. PRINT FINAL POLICY EVALUATION VALUES
# ============================================================

print("\n")
print("=" * 65)
print("POLICY EVALUATION CONVERGENCE")
print("=" * 65)

print(
    "Number of iterations:",
    convergence_iteration
)

print()

for state in states:

    print(
        f"{state}: {V_eval[state]:.4f}"
    )


# ============================================================
# 10. VALUE ITERATION
# ============================================================

print("\n")
print("=" * 65)
print("TASK 3 - VALUE ITERATION")
print("=" * 65)

# Initial value function
V = {
    state: 0.0
    for state in states
}

value_iteration_history = []

for iteration in range(1, 100):

    new_V = V.copy()

    delta = 0.0

    for state in states:

        # Goal is terminal
        if state == "G":

            new_V[state] = 0.0

            continue

        action_values = {}

        # Evaluate every possible action
        for action in actions:

            next_state, reward, done = step(
                state,
                action
            )

            if done:

                action_value = reward

            else:

                action_value = (
                    reward +
                    gamma * V[next_state]
                )

            action_values[action] = action_value

        # Select maximum action value
        best_value = max(
            action_values.values()
        )

        new_V[state] = best_value

        delta = max(
            delta,
            abs(best_value - V[state])
        )

    V = new_V

    value_iteration_history.append(
        (
            iteration,
            delta,
            V.copy()
        )
    )

    # Stop when converged
    if delta < THETA:

        break


# ============================================================
# 11. PRINT VALUE ITERATION RESULTS
# ============================================================

print()

print(
    f"{'Iteration':<12}"
    f"{'S1':<12}"
    f"{'S2':<12}"
    f"{'S3':<12}"
    f"{'S4':<12}"
    f"{'S5':<12}"
    f"{'Delta':<12}"
)

print("-" * 82)

for iteration, delta, values in value_iteration_history:

    print(
        f"{iteration:<12}"
        f"{values['S1']:<12.2f}"
        f"{values['S2']:<12.2f}"
        f"{values['S3']:<12.2f}"
        f"{values['S4']:<12.2f}"
        f"{values['S5']:<12.2f}"
        f"{delta:<12.4f}"
    )


# ============================================================
# 12. DERIVE OPTIMAL POLICY
# ============================================================

optimal_policy = {}

for state in states:

    # Goal has no action
    if state == "G":

        optimal_policy[state] = "-"

        continue

    best_action = None

    best_value = float("-inf")

    # Check all possible actions
    for action in actions:

        next_state, reward, done = step(
            state,
            action
        )

        if done:

            action_value = reward

        else:

            action_value = (
                reward +
                gamma * V[next_state]
            )

        if action_value > best_value:

            best_value = action_value

            best_action = action

    optimal_policy[state] = best_action


# ============================================================
# 13. TASK 3 OBSERVATION TABLE
# ============================================================

print("\n")
print("=" * 65)
print("OPTIMAL POLICY")
print("=" * 65)

print(
    f"{'State':<12}"
    f"{'Optimal Action':<20}"
    f"{'State Value':<15}"
)

print("-" * 47)

for state in states:

    if state == "G":

        print(
            f"{state:<12}"
            f"{'Terminal':<20}"
            f"{0.0:<15.2f}"
        )

    else:

        print(
            f"{state:<12}"
            f"{optimal_policy[state]:<20}"
            f"{V[state]:<15.2f}"
        )


# ============================================================
# 14. ARROW REPRESENTATION
# ============================================================

arrows = {
    "U": "↑",
    "D": "↓",
    "L": "←",
    "R": "→",
    "-": "G"
}

print("\n")
print("=" * 65)
print("OPTIMAL POLICY GRID")
print("=" * 65)

for row in grid:

    for state in row:

        if state == "G":

            print("[ G  ]", end=" ")

        else:

            print(
                f"[{state} {arrows[optimal_policy[state]]}]",
                end=" "
            )

    print()


# ============================================================
# 15. POLICY EXECUTION FUNCTION
# ============================================================

def execute_policy(
    policy_type,
    max_steps=20
):

    current_state = START_STATE

    path = [current_state]

    total_reward = 0

    for step_number in range(max_steps):

        # If goal already reached
        if current_state == "G":

            return (
                path,
                step_number,
                total_reward,
                True
            )

        # Select action according to policy
        if policy_type == "random":

            action = random.choice(actions)

        elif policy_type == "evaluated":

            action = given_policy[current_state]

        elif policy_type == "optimal":

            action = optimal_policy[current_state]

        else:

            raise ValueError(
                "Invalid policy type"
            )

        # Environment transition
        next_state, reward, done = step(
            current_state,
            action
        )

        # Update reward
        total_reward += reward

        # Move to next state
        current_state = next_state

        # Record path
        path.append(current_state)

        # Check terminal state
        if done:

            return (
                path,
                step_number + 1,
                total_reward,
                True
            )

    # Maximum number of steps reached
    return (
        path,
        max_steps,
        total_reward,
        False
    )


# ============================================================
# 16. RANDOM POLICY
# ============================================================

# Fixed seed makes the random experiment reproducible
random.seed(21)

random_path, random_steps, random_reward, random_goal = \
    execute_policy(
        "random",
        max_steps=20
    )

print("\n")
print("=" * 65)
print("RANDOM POLICY")
print("=" * 65)

print(
    "Path:",
    " -> ".join(random_path)
)

print(
    "Path Length:",
    random_steps
)

print(
    "Total Reward:",
    random_reward
)

print(
    "Goal Reached:",
    "Yes" if random_goal else "No"
)


# ============================================================
# 17. EVALUATED POLICY
# ============================================================

evaluated_path, evaluated_steps, evaluated_reward, evaluated_goal = \
    execute_policy(
        "evaluated",
        max_steps=20
    )

print("\n")
print("=" * 65)
print("EVALUATED POLICY")
print("=" * 65)

print(
    "Path:",
    " -> ".join(evaluated_path)
)

print(
    "Path Length:",
    evaluated_steps
)

print(
    "Total Reward:",
    evaluated_reward
)

print(
    "Goal Reached:",
    "Yes" if evaluated_goal else "No"
)


# ============================================================
# 18. OPTIMAL POLICY
# ============================================================

optimal_path, optimal_steps, optimal_reward, optimal_goal = \
    execute_policy(
        "optimal",
        max_steps=20
    )

print("\n")
print("=" * 65)
print("OPTIMAL POLICY")
print("=" * 65)

print(
    "Path:",
    " -> ".join(optimal_path)
)

print(
    "Path Length:",
    optimal_steps
)

print(
    "Total Reward:",
    optimal_reward
)

print(
    "Goal Reached:",
    "Yes" if optimal_goal else "No"
)


# ============================================================
# 19. TASK 4 FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 65)
print("TASK 4 - POLICY COMPARISON")
print("=" * 65)

print(
    f"{'Policy Type':<20}"
    f"{'Path Length':<15}"
    f"{'Total Reward':<15}"
    f"{'Goal Reached'}"
)

print("-" * 70)

print(
    f"{'Random Policy':<20}"
    f"{random_steps:<15}"
    f"{random_reward:<15}"
    f"{'Yes' if random_goal else 'No'}"
)

print(
    f"{'Evaluated Policy':<20}"
    f"{evaluated_steps:<15}"
    f"{evaluated_reward:<15}"
    f"{'Yes' if evaluated_goal else 'No'}"
)

print(
    f"{'Optimal Policy':<20}"
    f"{optimal_steps:<15}"
    f"{optimal_reward:<15}"
    f"{'Yes' if optimal_goal else 'No'}"
)


# ============================================================
# 20. SHORTEST PATH
# ============================================================

print("\n")
print("=" * 65)
print("SHORTEST PATH ANALYSIS")
print("=" * 65)

print(
    "Optimal Path:",
    " -> ".join(optimal_path)
)

print(
    "Minimum Number of Steps:",
    optimal_steps
)

print(
    "Optimal Total Reward:",
    optimal_reward
)

print(
    "Goal Reached:",
    "Yes" if optimal_goal else "No"
)