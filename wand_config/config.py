import wandb
from MODEL import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.config = {
    "learning_rate": 0.0001,
    "epochs": 2000,
    "batch_size": 10,
    "num_workers": 0,
    "weight_decay(l2)":0.0005,
}
# MAE
# criteria = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

# MSE
criteria = torch.nn.MSELoss()

#
c = 'MSELoss'
# criteria = torch.sqrt((x - y)**2).sum()

# criteria = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
wandb.login(key='6d9a6a365138fabaebdcb9377170090220a3ba00')
wandb.init(project="Simple_log_positive_Model", entity="maria_mikhalkova",config=wandb.config)
wandb.run.name = f" loss = {c}, lr={wandb.config['learning_rate']},epochs= {wandb.config['epochs']}, l2={wandb.config['weight_decay(l2)']} "

input_size = 407
hidden_size = 512
num_layers = 10  # 9
output_dim = 1

net = LSTMClassifier(input_size, hidden_size, num_layers, output_dim, device)
# net = predict_hours_net(input_size, hidden_size, num_layers, device)
optimizer = torch.optim.Adam(net.parameters(), lr=wandb.config['learning_rate'],
                             weight_decay=wandb.config['weight_decay(l2)'] )

epochs = wandb.config['epochs']
