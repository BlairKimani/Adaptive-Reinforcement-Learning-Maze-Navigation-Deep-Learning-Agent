# Adaptive-Reinforcement-Learning-Maze-Navigation-Deep-Learning-Agent
**MazeRL** is an interactive reinforcement-learning playground where an agent
taught itself to solve mazes — with no hand-coded pathfinding, no action
labels, and no shortcuts. Every move it makes comes from a 14-input Softmax
policy network trained end-to-end with the REINFORCE policy-gradient
algorithm. You can watch it fail, flail, and eventually find its way home,
while the live dashboard shows reward curves, action probabilities, and
per-episode statistics updating in real time.

The environment is a fully procedural grid maze: the generator carves a
guaranteed start→goal corridor, scatters obstacles, and validates
connectivity with BFS, so *every* map is solvable. Maps are decoupled from
the window, so the same code runs a 15×15 starter maze or a 1000×1000
labyrinth without resizing the Pygame surface — rendering is viewport-culled
to only the visible cells. An independent 2D camera supports smooth
follow-agent mode and a free-pan exploration mode, so you can scroll away
and inspect any part of the map while training keeps running in the
background. A minimap, clickable to teleport the camera, shows the full map,
visited cells, and the current viewport rectangle.

The RL pipeline is deliberately transparent. Each timestep the agent
receives a 14-dimensional observation (normalized position, relative goal
offset, orientation one-hot, four cardinal wall sensors, BFS-distance
heuristic, and its previous action) and the policy outputs a 4-way Softmax
distribution over relative actions — Forward, Right, Backward, Left.
During training it samples from this distribution; during evaluation it can
sample stochastically or act greedily. Rewards are shaped per-event
(+50 for the goal, ±1 for progress, −3 for wall bumps, −2 for revisits,
−0.1 per step, −10 on timeout), and after every episode the REINFORCE
update computes discounted returns, normalizes them into advantages, and
applies the entropy-regularized policy-gradient loss
`−log π(a|s)·A − β·H(π)` through a `tf.GradientTape` and Adam.
BFS is used *only* for map validation and reward shaping — it never
selects actions or supplies a route to the network.

Everything is persisted: model weights (with optimizer state for resuming
training), best policy checkpoints, procedural maps as JSON, best-ever
successful paths per map, full episode histories, and a per-episode training
CSV log. Save a map, close the app, reopen it, and the agent resumes on the
exact same environment. Switch to evaluation mode and test a trained policy
on unseen maps without touching its weights.

Built with Python 3.11+, Pygame, TensorFlow/Keras, and NumPy — two source
files (`Game.py` for environment, camera, UI, and training orchestration;
`model.py` for the network and gradient update), plus automatically created
directories for models, maps, paths, and logs. Run it with `python Game.py`.
