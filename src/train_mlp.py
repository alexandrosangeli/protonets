import pandas as pd

from mlp import MLP
from utils import mse_loss
import math

# TODO: Split into train/test sets and evaluate

data = pd.read_csv("../data/wine_quality/winequality-red.csv")

col_names = []
for k in data.iloc[0].keys()[0].split(';'):
    key = k.strip('"')
    col_name = "_".join([w for w in key.split(" ")])
    col_names.append(col_name)

df = pd.DataFrame(columns=col_names)

for i in range(len(data)):
    row = data.iloc[i]
    values = [float(x) for x in row.values[0].split(';')]
    values[-1] = values[-1] / 8 # normalise
    df.loc[i] = values


mlp = MLP(11, [3,3,1])
targets = []

size = len(df)
num_epochs = 4
batch_size = 50
total_steps = math.ceil(size / batch_size)
learning_rate = 0.3

for i in range(size):
    targets.append(df.iloc[i].values.flatten().tolist()[-1])


losses_after_epoch = []
for i in range(num_epochs):
    print(f"Epoch: [{i+1}/{num_epochs}]")
    counter = 0
    # iterate over the entire set
    for j in range(0, size, batch_size):
        # print(f"Step: [{counter+1}/{total_steps}]")
        counter += 1
        losses = []
        y_pred = []
        z = min(j+batch_size, size)
        # iterate over the batch
        for k in range(size)[j:z]:
            vals = df.iloc[k].values.flatten().tolist()
            input = vals[:-1]
            out = mlp(input)
            y_pred.append(out)


        mlp.zero_grad()
        loss = mse_loss(targets[j:z], y_pred)
        loss.backward()
        mlp.step(lr=learning_rate)
        losses.append(loss.data)

    losses_after_epoch.append(sum(losses) / total_steps)
    
for i in range(num_epochs):
    loss_ = losses_after_epoch[i]
    decrease = None
    if i > 0:
        decrease = (losses_after_epoch[i-1] - losses_after_epoch[i]) / losses_after_epoch[i-1]
        decrease *= 100
        print(f"Epoch {i+1} loss: {loss_}, Decrease : {decrease}%")
    else:
        print(f"Epoch {i+1} loss: {loss_}")