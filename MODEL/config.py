from MODEL import *
from enums import ModelType, CriteriaType
from typing import Optional


class Parameters:
    def __init__(self, config: Optional[dict] = None, model_type: ModelType = ModelType.Linear,
                 criteria_type: CriteriaType = CriteriaType.MAE) -> object:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criteria_type = criteria_type
        self._input_size = 407
        self._hidden_size = 512
        self._num_layers = 10  # 9
        self._output_dim = 1
        self.model_type = model_type
        self.huber_delta = 0.5

        if self.model_type is ModelType.LSTM:
            self.net = LSTMClassifier(self._input_size,
                                      self._hidden_size,
                                      self._num_layers,
                                      self._output_dim,
                                      self.device)
        elif self.model_type is ModelType.Linear:
            self.net = predict_hours_net(self._input_size,
                                         self._hidden_size,
                                         self._num_layers,
                                         self.device)
        elif self.model_type is ModelType.Linear_3MONTH:
            self.net = predict_hours_net_3MONTH(self._input_size,
                                                self._hidden_size,
                                                self._num_layers,
                                                self.device)

        if self.criteria_type is CriteriaType.MSE:
            self.criteria = torch.nn.MSELoss()
        elif self.criteria_type is CriteriaType.HuberLoss:
            self.criteria = torch.nn.HuberLoss(delta=self.huber_delta)
        elif self.criteria_type is CriteriaType.MAE:
            self.criteria = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

        self.optimizer = None
        self.epochs: int = 1000

        if config is not None:
            self.set_config(config)

    def set_config(self, config: Optional[dict]):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['learning_rate'],
                                          weight_decay=config['weight_decay(l2)'])
        self.epochs = config['epochs']



