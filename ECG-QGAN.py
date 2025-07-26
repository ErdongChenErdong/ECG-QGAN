import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader, Dataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import AdamW, Adam, RMSprop
import pywt
import scipy
import os
import pennylane as qml


class QBiGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_qubits=4,
                 n_qlayers=1,
                 n_vrotations=1,
                 batch_first=True,
                 return_sequences=False,
                 return_state=False,
                 backend="lightning.qubit"):
        super(QBiGRU, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = n_vrotations
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_reset = [f"wire_reset_{i}" for i in range(self.n_qubits)]
        self.wires_reset1 = [f"wire_reset1_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]

        self.wires_back_reset = [f"wire_back_reset_{i}" for i in range(self.n_qubits)]
        self.wires_back_reset1 = [f"wire_back_reset1_{i}" for i in range(self.n_qubits)]
        self.wires_back_update = [f"wire_back_update_{i}" for i in range(self.n_qubits)]

        self.dev_reset = qml.device(self.backend, wires=self.wires_reset)
        self.dev_reset1 = qml.device(self.backend, wires=self.wires_reset1)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)

        self.dev_back_reset = qml.device(self.backend, wires=self.wires_back_reset)
        self.dev_back_reset1 = qml.device(self.backend, wires=self.wires_back_reset1)
        self.dev_back_update = qml.device(self.backend, wires=self.wires_back_update)

        def ansatz(params, wires_type):
            # Entangling layer.
            for i in range(1, 3):
                for j in range(self.n_qubits):
                    if j + i < self.n_qubits:
                        qml.CNOT(wires=[wires_type[j], wires_type[j + i]])
                    else:
                        qml.CNOT(wires=[wires_type[j], wires_type[j + i - self.n_qubits]])

            # Variational layer.
            for i in range(self.n_qubits):
                qml.RX(params[0][i], wires=wires_type[i])

        #                 qml.RY(params[1][i], wires=wires_type[i])
        #                 qml.RZ(params[2][i], wires=wires_type[i])

        def VQC(features, weights, wires_type):
            # Preproccess input data to encode the initial state.
            # qml.templates.AngleEmbedding(features, wires=wires_type)
            qml.AmplitudeEmbedding(features, wires_type, normalize=True)
            #             ry_params = [torch.arctan(feature) for feature in features]
            #             rz_params = [torch.arctan(feature**2) for feature in features]
            #             for i in range(self.n_qubits):
            # #                 qml.Hadamard(wires=wires_type[i])
            #                 qml.RY(ry_params[i], wires=wires_type[i])
            #                 qml.RZ(ry_params[i], wires=wires_type[i])

            # Variational block.
            qml.layer(ansatz, self.n_qlayers, weights, wires_type=wires_type)

        def _circuit_reset(inputs, weights):
            VQC(inputs, weights, self.wires_reset)
            # qml.templates.AngleEmbedding(inputs, wires=self.wires_reset)
            # qml.templates.BasicEntanglerLayers(weights, wires=self.wires_reset)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_reset]

        self.qlayer_reset = qml.QNode(_circuit_reset, self.dev_reset, interface="torch")

        def _circuit_reset1(inputs, weights):
            VQC(inputs, weights, self.wires_reset1)
            # qml.templates.AngleEmbedding(inputs, wires=self.wires_reset1)
            # qml.templates.BasicEntanglerLayers(weights, wires=self.wires_reset1)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_reset1]

        self.qlayer_reset1 = qml.QNode(_circuit_reset1, self.dev_reset1, interface="torch")

        def _circuit_update(inputs, weights):
            VQC(inputs, weights, self.wires_update)
            # qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            # qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]

        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        def _circuit_back_reset(inputs, weights):
            VQC(inputs, weights, self.wires_back_reset)
            # qml.templates.AngleEmbedding(inputs, wires=self.wires_reset)
            # qml.templates.BasicEntanglerLayers(weights, wires=self.wires_reset)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_back_reset]

        self.qlayer_back_reset = qml.QNode(_circuit_back_reset, self.dev_back_reset, interface="torch")

        def _circuit_back_reset1(inputs, weights):
            VQC(inputs, weights, self.wires_back_reset1)
            # qml.templates.AngleEmbedding(inputs, wires=self.wires_reset1)
            # qml.templates.BasicEntanglerLayers(weights, wires=self.wires_reset1)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_back_reset1]

        self.qlayer_back_reset1 = qml.QNode(_circuit_back_reset1, self.dev_back_reset1, interface="torch")

        def _circuit_back_update(inputs, weights):
            VQC(inputs, weights, self.wires_back_update)
            # qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            # qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_back_update]

        self.qlayer_back_update = qml.QNode(_circuit_back_update, self.dev_back_update, interface="torch")

        weight_shapes = {"weights": (self.n_qlayers, self.n_vrotations, self.n_qubits)}
        print(
            f"weight_shapes = (n_qlayers, n_vrotations, n_qubits) = ({self.n_qlayers}, {self.n_vrotations}, {self.n_qubits})")
        # weight_shapes = {"weights": (n_qlayers, n_qubits)}
        # print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, 2 ** n_qubits)
        self.VQC = {
            'reset': qml.qnn.TorchLayer(self.qlayer_reset, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'reset1': qml.qnn.TorchLayer(self.qlayer_reset1, weight_shapes),
            'back_reset': qml.qnn.TorchLayer(self.qlayer_back_reset, weight_shapes),
            'back_update': qml.qnn.TorchLayer(self.qlayer_back_update, weight_shapes),
            'back_reset1': qml.qnn.TorchLayer(self.qlayer_back_reset1, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        # self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        back_hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)  # hidden state (output)
            back_h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t = init_states
            h_t = h_t[0]
            back_h_t = init_states
            back_h_t = back_h_t[0]

        # 正向传播
        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            r_t = torch.sigmoid(self.clayer_out(self.VQC['reset'](y_t)))  # reset block
            h_t1 = r_t * h_t
            v_t_1 = torch.cat((h_t1, x_t), dim=1)
            v_t1 = self.clayer_in(v_t_1)

            h_t2 = torch.tanh(self.clayer_out(self.VQC['reset1'](v_t1)))  # reset block
            z_t = torch.sigmoid(self.clayer_out(self.VQC['update'](y_t)))  # update block

            h_t3 = (1 - z_t) * h_t + z_t * h_t2

            hidden_seq.append(h_t3.unsqueeze(0))

        # 反向传播
        for t in reversed(range(seq_length)):
            # get features from the t-th element in seq, for all entries in the batch
            back_x_t = x[:, t, :]

            # Concatenate input and hidden state
            back_v_t = torch.cat((back_h_t, back_x_t), dim=1)

            # match qubit dimension
            back_y_t = self.clayer_in(back_v_t)

            back_r_t = torch.sigmoid(self.clayer_out(self.VQC['back_reset'](back_y_t)))  # reset block
            back_h_t1 = back_r_t * back_h_t
            back_v_t_1 = torch.cat((back_h_t1, back_x_t), dim=1)
            back_v_t1 = self.clayer_in(back_v_t_1)

            back_h_t2 = torch.tanh(self.clayer_out(self.VQC['back_reset1'](back_v_t1)))  # reset block
            back_z_t = torch.sigmoid(self.clayer_out(self.VQC['back_update'](back_y_t)))  # update block

            back_h_t3 = (1 - back_z_t) * back_h_t + back_z_t * back_h_t2

            back_hidden_seq.append(back_h_t3.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        back_hidden_seq = torch.cat(back_hidden_seq, dim=0)
        back_hidden_seq = back_hidden_seq.transpose(0, 1).contiguous()
        #         output = torch.cat([hidden_seq, back_hidden_seq], dim=-1)
        output = (hidden_seq + back_hidden_seq) / 2
        return output


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Generator, self).__init__()
        self.model = transfer_1
        self.bigru = QBiGRU(input_size, hidden_size)
        self.bn0 = nn.BatchNorm1d(256)


    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, 4, 25)
        x = self.model(x)
        output = self.bigru(x)
        out = output.reshape(batch_size, -1)
        # print(out.shape)
        out = self.bn0(out)
        # print(images.shape)
        return out