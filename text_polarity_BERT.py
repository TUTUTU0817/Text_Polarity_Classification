import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification

# 載入資料
data_df = pd.read_csv(".\\datasets\\train_2022.csv")

# 切割資料集
X = data_df['TEXT']
y = data_df['LABEL']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train_list = X_train.tolist()
X_test_list = X_test.tolist()


# 加載預訓練的Bert tokenizer和分類模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2)

# 使用tokenizer 將文本轉換為Bert輸入格式
X_train_encodings = tokenizer(X_train_list, truncation=True, padding=True)
X_test_encodings = tokenizer(X_test_list, truncation=True, padding=True)

# 訓練和微調模型
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train_encodings['input_ids']),
    torch.tensor(X_train_encodings['attention_mask']),
    torch.tensor(y_train.tolist()),
)

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_test_encodings['input_ids']),
    torch.tensor(X_test_encodings['attention_mask']),
    torch.tensor(y_test.tolist()),
)

# 定義dataloader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=8, shuffle=False)

# 將模型設置為訓練模式
model.train()

# 定義訓練迴圈
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(3):  # 進行三個 epoch 的訓練
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# 將模型設置為評估模式
model.eval()

# 進行測試並計算準確率
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(predictions == labels)
        total += len(labels)

accuracy = correct / total
print(f"Accuracy: {accuracy.item() * 100:.2f}%")


# 載入測試資料
test_data_df = pd.read_csv(".\\datasets\\test_no_answer_2022.csv")

# 使用 tokenizer 將新文本轉換為 BERT 輸入格式
new_encodings = tokenizer(test_data_df['TEXT'].tolist(
), truncation=True, padding=True, return_tensors='pt')

# 進行預測
with torch.no_grad():
    input_ids = new_encodings['input_ids'].to(model.device)
    attention_mask = new_encodings['attention_mask'].to(model.device)
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
test_data_df['LABEL'] = predictions
ans_bert = test_data_df.drop('TEXT', axis=1)
ans_bert.to_csv(
    '.\\ans_bert.csv', index=False)
