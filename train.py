import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from agent import DQN, ReplayMemory, Transition
from env import FraudDetectionEnv

# Veri
df = pd.read_excel('data/duzenli_veri.xlsx')
df = df.dropna(subset=['Class'])

X = df.drop('Class', axis=1)
y = df['Class']
X[['Time','Amount']] = StandardScaler().fit_transform(X[['Time','Amount']])

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Ortam başlatılır
env = FraudDetectionEnv(X_train, y_train)
n_features = X_train.shape[1]
n_actions  = 2
device     = torch.device('cpu')

policy_net = DQN(n_features, n_actions).to(device)
target_net = DQN(n_features, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory    = ReplayMemory(capacity=10000)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# ε-greedy parametreleri
eps_start, eps_end, eps_decay = 1.0, 0.05, 5000
steps_done = 0
gamma      = 0.99
batch_size = 64

# Kayıtlar
all_losses = []
all_rewards = []

# Eylem Fonksiyonu
def select_action(state):
    global steps_done
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if random.random() < eps_threshold:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        return policy_net(state_t).argmax().item()

# Model optimize edilir
def optimize_model():
    if len(memory) < batch_size:
        return None
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch      = torch.from_numpy(np.vstack(batch.state)).float().to(device)
    action_batch     = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(device)
    reward_batch     = torch.tensor(batch.reward, dtype=torch.float).unsqueeze(1).to(device)
    next_state_batch = torch.from_numpy(np.vstack(batch.next_state)).float().to(device)
    done_batch       = torch.tensor(batch.done, dtype=torch.float).unsqueeze(1).to(device)

    q_values   = policy_net(state_batch).gather(1, action_batch)
    next_q     = target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
    expected_q = reward_batch + (gamma * next_q * (1 - done_batch))

    loss = nn.functional.mse_loss(q_values, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Eğitim
num_episodes = 50
for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    losses = []

    while not done:
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        loss = optimize_model()
        if loss is not None:
            losses.append(loss)

    target_net.load_state_dict(policy_net.state_dict())
    avg_loss = np.mean(losses) if losses else 0.0
    all_losses.append(avg_loss)
    all_rewards.append(total_reward)
    print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Avg Loss = {avg_loss:.4f}")

# Grafikler
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(all_rewards)
plt.title("Episode Bazlı Toplam Ödül")
plt.xlabel("Episode")
plt.ylabel("Toplam Ödül")

plt.subplot(1, 2, 2)
plt.plot(all_losses)
plt.title("Episode Bazlı Ortalama Kayıp (Loss)")
plt.xlabel("Episode")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()

# Model kaydedilir
torch.save(policy_net.state_dict(), "model_dqn.pt")
print("✅ Eğitim tamamlandı. Model 'model_dqn.pt' olarak kaydedildi.")
