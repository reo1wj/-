# train.py

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datasets.dataset import build_dataset, RadioDataset
from models.cnn import CNNModel
import config


# 1️⃣ 数据
X, y = build_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_loader = DataLoader(
    RadioDataset(X_train, y_train),
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    RadioDataset(X_test, y_test),
    batch_size=config.BATCH_SIZE
)


# 2️⃣ 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel(config.NUM_CLASSES).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)


# 3️⃣ 训练
for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss={total_loss:.2f}")


# 4️⃣ 测试
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        pred = model(X_batch).argmax(1)
        total += y_batch.size(0)
        correct += (pred == y_batch).sum().item()

print("Accuracy:", correct / total)