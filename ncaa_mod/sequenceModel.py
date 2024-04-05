import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from fancyimpute import IterativeImputer as MICE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQUENCE_LENGTH = 3

data_paths = ['cleaned_csvs/cleaned-cbb2013.csv', 'cleaned_csvs/cleaned-cbb2014.csv', 'cleaned_csvs/cleaned-cbb2015.csv','cleaned_csvs/cleaned-cbb2016.csv', 
              'cleaned_csvs/cleaned-cbb2017.csv', 'cleaned_csvs/cleaned-cbb2018.csv', 'cleaned_csvs/cleaned-cbb2019.csv', 'cleaned_csvs/cleaned-cbb2020.csv', 
              'cleaned_csvs/cleaned-cbb2021.csv', 'cleaned_csvs/cleaned-cbb2022.csv', 'cleaned_csvs/cleaned-cbb2023.csv']

dataframes = {}
for path in data_paths:
    year = path.split('cleaned-cbb')[-1].split('.')[0]
    dataframes[year] = pd.read_csv(path)

for year, df in dataframes.items():
    df['Year'] = int(year)

combined_df = pd.concat(dataframes.values(), ignore_index=True)

postseason_non_nan = combined_df['POSTSEASON'].dropna().unique()
label_encoder = LabelEncoder()
combined_df['POSTSEASON_temp'] = combined_df['POSTSEASON'].fillna('NaN')
combined_df['POSTSEASON_encoded'] = label_encoder.fit_transform(combined_df['POSTSEASON_temp'])
combined_df.drop('POSTSEASON_temp', axis=1, inplace=True)
postseason_classes = label_encoder.classes_


nominal_columns = ['TEAM', 'CONF']
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(combined_df[nominal_columns])
onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(nominal_columns))
processed_df = pd.concat([combined_df.drop(nominal_columns, axis=1), onehot_encoded_df], axis=1)
processed_df.loc[(processed_df['Year'] == 2020) & (processed_df['POSTSEASON_encoded'] == label_encoder.transform(['NaN'])[0]), 'POSTSEASON_encoded'] = np.nan

X = processed_df.drop(['POSTSEASON', 'POSTSEASON_encoded', 'Year'], axis=1)
y = processed_df['POSTSEASON_encoded']

tscv = TimeSeriesSplit(n_splits=5)
scores = []

X_scaled_imputed = []
y_imputed = []

for train_index, test_index in tscv.split(X, groups=processed_df['Year']):
    X_train_val, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train_val, y_test = y.iloc[train_index], y.iloc[test_index]
    val_size = int(len(X_train_val) * 0.2)
    X_train, X_val = X_train_val[:-val_size], X_train_val[-val_size:]
    y_train, y_val = y_train_val[:-val_size], y_train_val[-val_size:]

    temporary_imputer = SimpleImputer(strategy="mean")
    y_train_temp_imputed = temporary_imputer.fit_transform(y_train.values.reshape(-1, 1))
    y_val_temp_imputed = temporary_imputer.transform(y_val.values.reshape(-1, 1))

    mice_imputer = MICE()
    y_train_imputed = mice_imputer.fit_transform(y_train_temp_imputed)
    y_val_imputed = mice_imputer.transform(y_val_temp_imputed)
    y_test_imputed = mice_imputer.transform(temporary_imputer.transform(y_test.values.reshape(-1, 1)))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_scaled_imputed.extend([X_train_scaled, X_val_scaled, X_test_scaled])
    y_imputed.extend([y_train_imputed.ravel(), y_val_imputed.ravel(), y_test_imputed.ravel()])
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled_imputed.extend([X_train_scaled, X_test_scaled])
    y_imputed.extend([y_train_imputed.ravel(), y_test_imputed.ravel()])
    
    X_train_sequences = []
    y_train_sequences = []
    for i in range(len(X_train_scaled) - SEQUENCE_LENGTH + 1):
        X_train_sequences.append(X_train_scaled[i:i + SEQUENCE_LENGTH])
        y_train_sequences.append(y_train_imputed[i + SEQUENCE_LENGTH - 1])

    X_val_sequences = []
    y_val_sequences = []
    for i in range(len(X_val_scaled) - SEQUENCE_LENGTH + 1):
        X_val_sequences.append(X_val_scaled[i:i + SEQUENCE_LENGTH])
        y_val_sequences.append(y_val_imputed[i + SEQUENCE_LENGTH - 1])

    X_test_sequences = []
    y_test_sequences = []
    for i in range(len(X_test_scaled) - SEQUENCE_LENGTH + 1):
        X_test_sequences.append(X_test_scaled[i:i + SEQUENCE_LENGTH])
        y_test_sequences.append(y_test_imputed[i + SEQUENCE_LENGTH - 1])

    X_train_tensor = torch.tensor(X_train_sequences, dtype=torch.float32, device=DEVICE)
    y_train_tensor = torch.tensor(y_train_sequences, dtype=torch.long, device=DEVICE)
    X_val_tensor = torch.tensor(X_val_sequences, dtype=torch.float32, device=DEVICE)
    y_val_tensor = torch.tensor(y_val_sequences, dtype=torch.long, device=DEVICE)
    X_test_tensor = torch.tensor(X_test_sequences, dtype=torch.float32, device=DEVICE)
    y_test_tensor = torch.tensor(y_test_sequences, dtype=torch.long, device=DEVICE)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        o = self.linear(y[:, :, -1])
        return F.log_softmax(o, dim=1)

input_size = X_train_tensor.shape[2]
output_size = len(postseason_classes)
num_channels = [32, 64, 128]
kernel_size = 2
dropout = 0.2
model = TCN(input_size, output_size, num_channels, kernel_size, dropout).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50
batch_size = 32

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_acc += (predicted == batch_y).sum().item()

    val_loss /= len(val_loader)
    val_acc /= len(val_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    y_pred = predicted.cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=postseason_classes)
    cm = confusion_matrix(y_true, y_pred)
    scores.append(accuracy)
    print(f"Fold {len(scores)}: Accuracy = {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)