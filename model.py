import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
STATE_DIM: int = 14
ACTION_DIM: int = 4


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def create_policy_model(
    state_dim: int = STATE_DIM,
    learning_rate: float = 0.001,
    hidden_units=(128, 64),
    ):
    """
    Creates a neural network that outputs a softmax probability distribution
    across 4 discrete actions:
        0 = forward, 1 = right, 2 = backward, 3 = left

    Parameters
    ----------
    state_dim      : number of input features (must be 14 for this game)
    learning_rate  : Adam learning rate
    hidden_units   : tuple of hidden-layer sizes

    Returns
    -------
    Compiled tf.keras.Model (softmax policy).
    """
    if state_dim <= 0:
        raise ValueError("state_dim must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    inputs = layers.Input(shape=(state_dim,), dtype=tf.float32, name="state")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(int(units), activation="relu", name=f"hidden_{i}")(x)
    outputs = layers.Dense(ACTION_DIM, activation="softmax", name="policy")(x)

    model = tf.keras.Model(inputs, outputs, name="policy_agent")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
    )
    return model


def create_optimizer(learning_rate: float = 0.001):
    """Standalone Adam optimizer (so its state can be saved/restored)."""
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


# --------------------------------------------------------------------------- #
# REINFORCE training step
# --------------------------------------------------------------------------- #
def train_policy_step(
    model,
    optimizer,
    states,
    actions,
    rewards,
    gamma=0.99,
    entropy_coef=0.05,
    ):
    """
    Trains the policy model using the REINFORCE algorithm on one episode.

    Parameters
    ----------
    model         : Keras policy model (softmax over 4 actions)
    optimizer     : tf.keras optimizer (e.g. Adam)
    states        : sequence of 1-D state vectors (length N)
    actions       : sequence of ints in [0, 3]       (length N)
    rewards       : sequence of floats               (length N)
    gamma         : discount factor
    entropy_coef  : weight of entropy regularization

    Returns
    -------
    (policy_loss, entropy, episode_return) : tuple of floats
        Returns (0.0, 0.0, 0.0) for empty trajectories.
    """
    n = len(states)
    if n == 0 or len(actions) != n or len(rewards) != n:
        return 0.0, 0.0, 0.0

    states_np  = np.asarray(states,  dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.int32)
    rewards_np = np.asarray(rewards, dtype=np.float32)

    if states_np.shape[1] != model.input_shape[-1]:
        raise ValueError(
            f"State dim mismatch: got {states_np.shape[1]}, "
            f"expected {model.input_shape[-1]}"
        )
    if np.any(actions_np < 0) or np.any(actions_np >= ACTION_DIM):
        raise ValueError("actions must be integers in [0, 3]")

    # ---------------- 1. Discounted returns & normalization ---------------- #
    returns = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = float(rewards_np[t]) + gamma * running
        returns[t] = running

    if n > 1 and float(returns.std()) > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # ---------------- 2. Batched policy-gradient loss ---------------------- #
    states_tensor  = tf.convert_to_tensor(states_np,  dtype=tf.float32)
    actions_tensor = tf.convert_to_tensor(actions_np, dtype=tf.int32)
    returns_tensor = tf.convert_to_tensor(returns,    dtype=tf.float32)

    with tf.GradientTape() as tape:
        probs = model(states_tensor, training=True)              # (N, 4)
        probs = tf.clip_by_value(probs, 1e-8, 1.0)                # avoid log(0)

        one_hot = tf.one_hot(actions_tensor, ACTION_DIM, dtype=tf.float32)
        log_prob_selected = tf.reduce_sum(tf.math.log(probs) * one_hot, axis=1)

        # REINFORCE: L = -E[ log π(a|s) * G_t ]
        policy_loss = -tf.reduce_mean(log_prob_selected * returns_tensor)

        # Entropy: H = -Σ p log p
        entropy_per_state = -tf.reduce_sum(probs * tf.math.log(probs), axis=1)
        mean_entropy = tf.reduce_mean(entropy_per_state)

        total_loss = policy_loss - entropy_coef * mean_entropy

    # ---------------- 3. Apply gradients ----------------------------------- #
    grads = tape.gradient(total_loss, model.trainable_variables)
    grads = [
        g if g is not None else tf.zeros_like(v)
        for g, v in zip(grads, model.trainable_variables)
    ]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return (
        float(policy_loss.numpy()),
        float(mean_entropy.numpy()),
        float(rewards_np.sum()),
    )


# --------------------------------------------------------------------------- #
# Action-selection helpers (used by Game.py)
# --------------------------------------------------------------------------- #
def action_probabilities(model, state):
    """Return a length-4 softmax probability vector for one state."""
    s = np.asarray(state, dtype=np.float32).reshape(1, -1)
    probs = model(s, training=False).numpy().reshape(-1)
    probs = np.clip(probs, 1e-8, 1.0)
    probs /= probs.sum()
    return probs.astype(np.float32)


def sample_action(probs, rng):
    """Stochastic action selection for training."""
    return int(rng.choice(ACTION_DIM, p=probs))


def greedy_action(probs, rng=None, epsilon=0.0):
    """
    Deterministic action selection for evaluation, with optional 
    epsilon-greedy randomness to escape wall traps.
    
    Parameters
    ----------
    probs   : length-4 probability array
    rng     : numpy random number generator instance (np.random.default_rng())
    epsilon : probability of taking a completely random action (e.g., 0.1 for 10%)
    """
    if epsilon > 0.0 and rng is not None and rng.random() < epsilon:
        # Inject random exploration to break out of wall loops
        return int(rng.integers(0, ACTION_DIM))
    
    # Default behavior: choose the highest probability action
    return int(np.argmax(probs))

# --------------------------------------------------------------------------- #
# Smoke-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = create_policy_model(STATE_DIM, learning_rate=0.005)
    optimizer = create_optimizer(0.005)

    # Tiny synthetic episode.
    rng = np.random.default_rng(0)
    s = rng.normal(size=(10, STATE_DIM)).astype(np.float32)
    a = rng.integers(0, 4, size=10).tolist()
    r = rng.normal(size=10).astype(np.float32).tolist()

    loss, ent, ret = train_policy_step(model, optimizer, s, a, r)
    print(f"loss={loss:.4f}  entropy={ent:.4f}  return={ret:.4f}")
    print("probs:", action_probabilities(model, s[0]))
