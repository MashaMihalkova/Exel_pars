import torch
import torch.nn as nn
import torch.cuda


class positive_weights_linear(nn.Module):
    """
        class with positive weights output
        input array (shape (373) ) and res_id (torch.int64 or torch.long)
    """

    def __init__(self, in_features, out_features):
        super(positive_weights_linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_x, res_id):
        return torch.matmul(self.weight[res_id.long()].exp(), input_x)


class predict_hours_net(nn.Module):
    """
        всего работ по нормам 373 шт. = "in_features" в слое "activity_dense"
        всего видов техники = 246 (все виды техники берутся из норм PO№№№ ) = "out_features" в слое "activity_dense"
        всего подрядчиков 4 шт. = "in_features" в слое "contractor_dense"
        всего проектов 23 шт. = "in_features" в слое "proj_dense"
    """

    def __init__(self, input_size, hidden_size, num_layers, device):
        super(predict_hours_net, self).__init__()
        # self.num_layers = num_layers
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.device = device
        # self.LSTM2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
        #                      dropout=0.2)

        #  out_features = количество техник 246
        self.activity_dense = positive_weights_linear(in_features=373, out_features=246)
        self.proj_dense = torch.nn.Linear(in_features=23, out_features=1, bias=False)
        self.contractor_dense = torch.nn.Linear(in_features=4, out_features=1, bias=False)
        self.year_dense = torch.nn.Linear(in_features=2, out_features=1, bias=True)
        self.month_dense = torch.nn.Linear(in_features=12, out_features=1, bias=False)

    def forward(self, x):
        # умножили на матрицу весов
        # sum_of_activities = self.activity_dense(x[2:-4], x[-1].long())  # если трекинг!!!
        sum_of_activities = self.activity_dense(x[2:-3], x[-1].long())

        # MULTIPLY BY PROJ
        # sum_of_proj = self.proj_dense.weight[0, x[0].long()] * sum_of_activities

        # умножили на коэф месяца
        sum_of_activities_month = self.month_dense.weight[0, x[-3].long()] * sum_of_activities

        # умножили на коэф.года
        sum_of_activities_month_year = self.year_dense.weight[0, x[-2].long()] * sum_of_activities_month

        # умножили на коэф.контрактора
        predict = self.contractor_dense.weight[0, x[1].long()] * sum_of_activities_month_year
        #
        # h_1 = Variable(torch.zeros(
        #     self.num_layers, x.size(0), self.hidden_size).to(self.device))
        # c_1 = Variable(torch.zeros(
        #     self.num_layers, x.size(0), self.hidden_size).to(self.device))
        # _, (hn, cn) = self.LSTM2(x, (h_1, c_1))
        # y = hn.view(-1, self.hidden_size)
        # final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]

        return predict


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device = device
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(self.device) for t in (h0, c0)]


class predict_hours_net_3MONTH(nn.Module):
    """
        всего работ по нормам 373 шт. = "in_features" в слое "activity_dense"
        всего видов техники = 246 (все виды техники берутся из норм PO№№№ ) = "out_features" в слое "activity_dense"
        всего подрядчиков 4 шт. = "in_features" в слое "contractor_dense"
        всего проектов 23 шт. = "in_features" в слое "proj_dense"
        последние три ячейки это результат работы техники за последние 3 месяца
    """

    def __init__(self, input_size, hidden_size, num_layers, device):
        super(predict_hours_net_3MONTH, self).__init__()
        # todo: контракторов 4 , в статистику заносятся их ид который просто чисто (например 14)

        #  out_features = количество техник 246
        self.activity_dense = positive_weights_linear(in_features=373, out_features=246)

        self.proj_dense = torch.nn.Linear(in_features=23, out_features=1, bias=False)

        self.contractor_dense = torch.nn.Linear(in_features=4, out_features=1, bias=False)

        self.year_dense = torch.nn.Linear(in_features=2, out_features=1, bias=False)

        self.month_dense = torch.nn.Linear(in_features=12, out_features=1, bias=False)

        self.last_3_month = torch.nn.Linear(in_features=3, out_features=1, bias=False)

    def forward(self, x):
        month = x[-6].long() - 1   # так как выделенный слой имеет 12 весов-один вес для каждого месяца,начиная с 0-го
        year = x[-5].long()
        res_id = x[-4].long()
        last_3_month = x[-3:]
        contr_id = x[1].long()

        sum_of_activities = self.activity_dense(x[2:-6], res_id)
        sum_of_month = torch.sum(torch.relu(self.last_3_month.weight[0] * last_3_month))

        sum_of_activities_month = self.month_dense.weight[0, month] * (sum_of_activities + sum_of_month)

        sum_of_activities_month_year = self.year_dense.weight[0, year] * sum_of_activities_month

        predict = self.contractor_dense.weight[0, contr_id] * sum_of_activities_month_year

        return predict

