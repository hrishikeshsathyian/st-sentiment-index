import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle


# Load the embeddings
print("\n📂 Loading data...")
embeddings_df = pd.read_pickle('/Users/hrishikeshsathyian/Desktop/nlp-insa/dataset/daily_embeddings.pkl')
sti_df = pd.read_csv('/Users/hrishikeshsathyian/Desktop/nlp-insa/dataset/sti_historical_data.csv')

# Clean dates
sti_df['Date'] = pd.to_datetime(sti_df['Date']).dt.date
embeddings_df['date'] = pd.to_datetime(embeddings_df['date']).dt.date

# Merge and sort by date 
merged_df = embeddings_df.merge(sti_df, left_on='date', right_on='Date', how='inner')
merged_df = merged_df.sort_values('date').reset_index(drop=True)
print(f"Total days: {len(merged_df)}")

# Create target: 1 if price went up (Close > Open), 0 if down
merged_df['target'] = (merged_df['Close'] > merged_df['Open']).astype(int)

print(f"\nUp days: {merged_df['target'].sum()} ({merged_df['target'].mean()*100:.1f}%)")
print(f"Down days: {(~merged_df['target'].astype(bool)).sum()} ({(1-merged_df['target'].mean())*100:.1f}%)")

# ========================================
# Traditional ML Models (Same Day)
# ========================================
print("\n" + "="*80)
print("PART 1: TRADITIONAL ML MODELS (SAME-DAY PREDICTION)")
print("="*80)

# Prepare features
X_title = np.vstack(merged_df['title_embedding'].values)
X_summary = np.vstack(merged_df['summary_embedding'].values)
X = np.hstack([X_title, X_summary])  # 768 features
y = merged_df['target'].values

print(f"\nFeatures: {X.shape}")
print(f"Target: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# Model 1: Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_test_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_test_pred)
print(f"LR Test Accuracy: {lr_acc:.4f}")

# Model 2: Random Forest
print("\n Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
rf_test_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_test_pred)
print(f"RF Test Accuracy: {rf_acc:.4f}")

# Model 3: XGBoost
print("\n Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_test_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_test_pred)
print(f"XGB Test Accuracy: {xgb_acc:.4f}")

# Show detailed results for XGBoost
print("\n" + "="*80)
print("XGBOOST - DETAILED RESULTS")
print("="*80)
print(classification_report(y_test, xgb_test_pred, target_names=['Down', 'Up']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_test_pred))

# ========================================
# PART 2: LSTM Time Series Model
# ========================================
print("\n" + "="*80)
print("PART 2: LSTM TIME SERIES MODEL (SEQUENCE-BASED)")
print("="*80)

# Create sequences: use past N days to predict current day
SEQUENCE_LENGTH = 5  # Use past 5 days of news

def create_sequences(embeddings, targets, seq_length):
    """Create sequences for LSTM"""
    X_seq, y_seq = [], []
    for i in range(seq_length, len(embeddings)):
        X_seq.append(embeddings[i-seq_length:i])
        y_seq.append(targets[i])
    return np.array(X_seq), np.array(y_seq)

print(f"\nCreating sequences with length {SEQUENCE_LENGTH}...")
X_seq, y_seq = create_sequences(X, y, SEQUENCE_LENGTH)
print(f"Sequence shape: {X_seq.shape}")  # (samples, sequence_length, 768)

# Split for time series (no shuffle to preserve temporal order!)
train_size = int(0.8 * len(X_seq))
X_train_seq, X_test_seq = X_seq[:train_size], X_seq[train_size:]
y_train_seq, y_test_seq = y_seq[:train_size], y_seq[train_size:]

print(f"Train sequences: {len(X_train_seq)} | Test sequences: {len(X_test_seq)}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_seq)
y_train_tensor = torch.LongTensor(y_train_seq)
X_test_tensor = torch.FloatTensor(X_test_seq)
y_test_tensor = torch.LongTensor(y_test_seq)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=2, num_classes=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out

# Initialize model
print("\nTraining LSTM...")
lstm_model = LSTMModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    lstm_model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}")

# Evaluate LSTM
lstm_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = lstm_model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

lstm_acc = accuracy_score(all_labels, all_preds)
print(f"\nLSTM Test Accuracy: {lstm_acc:.4f}")

print("\n" + "="*80)
print("LSTM - DETAILED RESULTS")
print("="*80)
print(classification_report(all_labels, all_preds, target_names=['Down', 'Up']))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ========================================
# SUMMARY
# ========================================
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)
print(f"Logistic Regression: {lr_acc:.4f}")
print(f"Random Forest:       {rf_acc:.4f}")
print(f"XGBoost:             {xgb_acc:.4f}")
print(f"LSTM (Time Series):  {lstm_acc:.4f}")

