import pandas as pd
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from agent import DQN
from env import FraudDetectionEnv

# ——— 1. VERİYİ YÜKLE ———
df = pd.read_excel('data/duzenli_veri.xlsx')
df = df.dropna(subset=['Class'])

X = df.drop('Class', axis=1)
y = df['Class']
X[['Time','Amount']] = StandardScaler().fit_transform(X[['Time','Amount']])

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# ——— 2. MODELİ YÜKLE ———
device = torch.device('cpu')
n_features = X_test.shape[1]
n_actions  = 2

policy_net = DQN(n_features, n_actions).to(device)
policy_net.load_state_dict(torch.load("model_dqn.pt", map_location=device))
policy_net.eval()
print("✅ Eğitilmiş model yüklendi.")

# ——— 3. TEST ———
env_test = FraudDetectionEnv(X_test, y_test)
state = env_test.reset()
preds, trues = [], []
done = False

while not done:
    with torch.no_grad():
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = policy_net(state_t).argmax().item()
    next_state, _, done, info = env_test.step(action)
    preds.append(action)
    trues.append(info['true_label'])
    state = next_state

# ——— 4. METRİKLERİ YAZDIR ———
report = classification_report(trues, preds, digits=4)
roc_auc = roc_auc_score(trues, preds)
accuracy = accuracy_score(trues, preds)
print("\n📊 Test Sonuçları:")
print(report)
print("ROC AUC:", roc_auc)
print(f"Accuracy: {accuracy:.4f} → %{accuracy * 100:.2f}")

# ——— 5. TKINTER ARAYÜZ ———
df_results = X_test.copy()
df_results["pred"] = preds
df_results["true"] = trues

columns_to_show = ["Time", "Amount", "pred", "true"]

root = tk.Tk()
root.title("Fraud Detection - Test Arayüzü")
root.geometry("750x700")

# Başlık
label = tk.Label(root, text="Test Sonuçları", font=("Arial", 16, "bold"))
label.pack(pady=5)

# Doğruluk
acc_label = tk.Label(root, text=f"Doğruluk: %{accuracy * 100:.2f}", font=("Arial", 12))
acc_label.pack()

# Tablo
tree = ttk.Treeview(root, columns=columns_to_show, show="headings", height=18)
for col in columns_to_show:
    tree.heading(col, text=col)
    tree.column(col, width=100)

tree.pack()

# Yanlış tahminleri kırmızı göster
tree.tag_configure("wrong", background="misty rose")

# Gösterme fonksiyonları
def show_all():
    tree.delete(*tree.get_children())
    for _, row in df_results.iterrows():
        vals = [row[c] for c in columns_to_show]
        is_wrong = int(row["pred"]) != int(row["true"])
        tag = "wrong" if is_wrong else ""
        tree.insert("", tk.END, values=vals, tags=(tag,))

def show_frauds():
    tree.delete(*tree.get_children())
    for _, row in df_results.iterrows():
        if row["pred"] == 1:
            vals = [row[c] for c in columns_to_show]
            is_wrong = int(row["pred"]) != int(row["true"])
            tag = "wrong" if is_wrong else ""
            tree.insert("", tk.END, values=vals, tags=(tag,))

# Butonlar
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

btn_all = tk.Button(btn_frame, text="Tümünü Göster", command=show_all)
btn_all.pack(side=tk.LEFT, padx=10)

btn_fraud = tk.Button(btn_frame, text="Sadece Fraud", command=show_frauds)
btn_fraud.pack(side=tk.LEFT, padx=10)

# Test sonuçlarını gösteren metin kutusu
txt_frame = tk.LabelFrame(root, text="📊 Test Raporu", font=("Arial", 10, "bold"))
txt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

txt = scrolledtext.ScrolledText(txt_frame, wrap=tk.WORD, height=10, font=("Consolas", 10))
txt.pack(fill=tk.BOTH, expand=True)

txt.insert(tk.END, report + "\n")
txt.insert(tk.END, f"ROC AUC: {roc_auc:.4f}\n")
txt.insert(tk.END, f"Accuracy: %{accuracy * 100:.2f}\n")
txt.config(state="disabled")

# Açılışta tümünü göster
show_all()
root.mainloop()
