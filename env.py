import numpy as np
import pandas as pd

class FraudDetectionEnv:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.n_samples, self.n_features = self.X.shape
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.X.iloc[0].values.astype(np.float32)

    def step(self, action: int):
        label = self.y.iloc[self.current_step]

        # Güncellenmiş ödül sistemi
        if action == 1:  # 'fraud' dendi
            if label == 1:
                reward = 5.0     # ✅ Doğru fraud → yüksek ödül
            else:
                reward = -1.0    # ❌ Yanlış alarm → küçük ceza
        else:  # 'normal' dendi
            if label == 1:
                reward = -5.0    # ❌ Fraud kaçtı → ağır ceza
            else:
                reward = +0.5    # ✅ Doğru normal → az ödül

        self.current_step += 1
        done = self.current_step >= self.n_samples

        next_state = (
            self.X.iloc[self.current_step].values.astype(np.float32)
            if not done else np.zeros(self.n_features, dtype=np.float32)
        )

        return next_state, reward, done, {'true_label': label}
