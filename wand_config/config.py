import wandb
from MODEL import *
from enums import ModelType, CriteriaType


def WandB():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE: int = 8
    LR: float = 0.001
    EPOCH: int = 500
    NW: int = 0
    L2: float = 0.0001
    flag = 0
    error_flag = 0
    wandb.config = {
        "learning_rate": LR,
        "epochs": EPOCH,
        "batch_size": BATCH_SIZE,
        "num_workers": NW,
        "weight_decay(l2)": L2,

    }

    # wandb.config = {
    #     "learning_rate": 0.0001,
    #     "epochs": 2000,
    #     "batch_size": 10,
    #     "num_workers": 0,
    #     "weight_decay(l2)":0.0005,
    # }
    # MAE
    # criteria = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

    # MSE
    criteria = torch.nn.MSELoss()
    model_type = ModelType.Linear
    criteria_type = CriteriaType.HuberLoss
    #
    # if CriteriaType.MSE:
    # c = CriteriaType.HuberLoss
    c = 'HuberLoss'
    # criteria = torch.sqrt((x - y)**2).sum()

    # criteria = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    wandb.login(key='6d9a6a365138fabaebdcb9377170090220a3ba00')
    wandb.init(project="tracking_mechanizm", entity="maria_mikhalkova", config=wandb.config)
    wandb.run.name = f" loss = {c}, lr={wandb.config['learning_rate']},epochs= {wandb.config['epochs']}, " \
                     f"l2={wandb.config['weight_decay(l2)']} "
    #
    # input_size = 407
    # hidden_size = 512
    # num_layers = 10  # 9
    # output_dim = 1

    # net = LSTMClassifier(input_size, hidden_size, num_layers, output_dim, device)
    # net = predict_hours_net(input_size, hidden_size, num_layers, device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=wandb.config['learning_rate'],
    #                              weight_decay=wandb.config['weight_decay(l2)'] )

    epochs = wandb.config['epochs']
