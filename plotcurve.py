import numpy as np
import matplotlib.pyplot as plt

ev = np.load("ev_log_2505_6.npy")
ev_steps = np.load("ev_timesteps_2505_6.npy")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

rewards = np.load("reward_log_2505_6.npy")
window = min(50, len(rewards) // 2)
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
ax1.plot(rewards, alpha=0.3, color='steelblue')
ax1.plot(range(window-1, len(rewards)), smoothed, color='steelblue', linewidth=2)
ax1.set_ylabel("Episode reward")
ax1.set_xlabel("Episode")
ax1.grid(True, alpha=0.3)

ax2.plot(ev_steps, ev, color='orange', linewidth=2)
ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (1.0)')
ax2.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Random (0.0)')
ax2.set_ylabel("Explained variance")
ax2.set_xlabel("Timesteps")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_stats_2505_5.png", dpi=150)
plt.show()