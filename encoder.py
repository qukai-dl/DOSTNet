import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data import TransportSupport


class CausalHistoryEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, history_steps: int):
        super().__init__()
        layers = []
        receptive_field = 1
        dilation = 1
        channels = in_dim
        while receptive_field < history_steps or not layers:
            layers.append(nn.Conv1d(channels, hidden_dim, 3, dilation=dilation))
            channels = hidden_dim
            receptive_field += 2 * dilation
            dilation *= 2
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = torch.tanh(layer(F.pad(x, (2 * layer.dilation[0], 0))))
        return x.transpose(1, 2)


class DOSTNetCore(nn.Module):
    def __init__(
        self,
        history_steps: int,
        history_feature_dim: int,
        nwp_feature_dim: int,
        static_feature_dim: int,
        hidden_dim: int = 64,
        edge_feature_dim: int = 9,
        face_feature_dim: int = 10,
        beta_max: float = 0.8,
        eta_max: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta_max = beta_max
        self.eta_max = eta_max
        self.history_encoder = CausalHistoryEncoder(
            history_feature_dim + static_feature_dim,
            hidden_dim,
            history_steps,
        )
        known_dim = nwp_feature_dim + static_feature_dim + 1
        self.local = nn.Linear(hidden_dim + known_dim, hidden_dim)
        self.transfer_state = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.transfer_edge = nn.Linear(edge_feature_dim, hidden_dim)
        self.route = nn.Linear(3 * hidden_dim, hidden_dim)
        self.context = nn.Linear(3 * hidden_dim, hidden_dim)
        self.beta_gate = nn.Linear(known_dim, 1)
        self.gamma_gate = nn.Linear(known_dim, 1)
        self.eta_gate = nn.Linear(face_feature_dim, 1)

    def _transfer(self, state: Tensor, features: Tensor) -> Tensor:
        return torch.tanh(self.transfer_state(state) + self.transfer_edge(features))

    def forward(
        self,
        history_features: Tensor,
        future_nwp: Tensor,
        static_features: Tensor,
        support: TransportSupport,
    ) -> Tensor:
        m_farms, history_steps, _ = history_features.shape
        forecast_steps = future_nwp.shape[1]
        static_history = static_features[:, None, :].expand(-1, history_steps, -1)
        encoded_history = self.history_encoder(
            torch.cat([history_features, static_history], dim=-1)
        )
        states = {
            (i, s): encoded_history[i, s + history_steps - 1]
            for i in range(m_farms)
            for s in range(-history_steps + 1, 1)
        }
        edge_features = [
            torch.as_tensor(
                edge.features,
                dtype=history_features.dtype,
                device=history_features.device,
            )
            for edge in support.edges
        ]
        face_features = [
            torch.as_tensor(
                face.features,
                dtype=history_features.dtype,
                device=history_features.device,
            )
            for face in support.faces
        ]
        for f in range(1, forecast_steps + 1):
            lead = history_features.new_tensor([f / forecast_steps])
            for j in range(m_farms):
                known = torch.cat([future_nwp[j, f - 1], static_features[j], lead])
                local = torch.tanh(self.local(torch.cat([states[(j, 0)], known])))
                edge_ids = support.incoming_edges.get((j, f), [])
                if edge_ids:
                    weights = history_features.new_tensor(
                        [support.edges[k].score for k in edge_ids]
                    )
                    weights = weights / weights.sum()
                    graph = torch.stack(
                        [
                            weights[n]
                            * self._transfer(
                                states[support.edges[k].source], edge_features[k]
                            )
                            for n, k in enumerate(edge_ids)
                        ]
                    ).sum(dim=0)
                else:
                    graph = torch.zeros_like(local)
                face_ids = support.incoming_faces.get((j, f), [])
                if face_ids:
                    weights = history_features.new_tensor(
                        [support.faces[k].score for k in face_ids]
                    )
                    weights = weights / weights.sum()
                    messages = []
                    for n, k in enumerate(face_ids):
                        face = support.faces[k]
                        h_a = states[face.source]
                        h_b = states[face.middle]
                        u = self._transfer(h_a, edge_features[face.first_edge])
                        pure = self._transfer(u, edge_features[face.second_edge])
                        middle = self._transfer(h_b, edge_features[face.second_edge])
                        direct = self._transfer(h_a, edge_features[face.direct_edge])
                        route = torch.tanh(
                            self.route(torch.cat([pure, direct, pure - direct]))
                        )
                        context = torch.tanh(
                            self.context(torch.cat([middle, pure, middle - pure]))
                        )
                        eta = self.eta_max * torch.sigmoid(self.eta_gate(face_features[k]))
                        messages.append(weights[n] * ((1.0 - eta) * route + eta * context))
                    face_message = torch.stack(messages).sum(dim=0)
                    gamma = torch.sigmoid(self.gamma_gate(known))
                else:
                    face_message = torch.zeros_like(local)
                    gamma = history_features.new_zeros(1)
                if edge_ids:
                    beta = self.beta_max * torch.sigmoid(self.beta_gate(known))
                else:
                    beta = history_features.new_zeros(1)
                transport = (1.0 - gamma) * graph + gamma * face_message
                states[(j, f)] = (1.0 - beta) * local + beta * transport
        return torch.stack(
            [
                torch.stack([states[(j, f)] for f in range(1, forecast_steps + 1)])
                for j in range(m_farms)
            ]
        )

    @torch.no_grad()
    def project_nonexpansive_weights(self) -> None:
        def project(weight: Tensor, limit: float) -> None:
            row_sum = weight.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
            weight.mul_(torch.clamp(limit / row_sum, max=1.0))

        project(self.transfer_state.weight, 1.0)
        project(self.route.weight, 0.5)
        project(self.context.weight, 0.5)
