import numpy as np
import torch
import torch.nn.functional as F
from rlpyt.agents.base import AgentStep
from rlpyt.agents.pg.base import AgentInfo
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.distributions.categorical import DistInfo
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

DEFAULT_HIDDEN_LAYERS = [64, 64]


class CustomAgent(CategoricalPgAgent):
    def eval_step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        pi, value = self.model(*model_inputs)
        dist_info = DistInfo(prob=pi)
        # action = self.distribution.sample(dist_info)
        action = torch.argmax(pi)
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)


# mixin is to convert env specs to model parameters
class InsertionEnvMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)


class StructuredInsertionEnvMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        sub_indices = env_spaces.action.space.sub_indices
        n = env_spaces.action.n
        return dict(observation_shape=env_spaces.observation.shape,
                    n_actions=n,
                    sub_indices=sub_indices)


class PPOStructuredInsertionModel(torch.nn.Module):
    """Implement the policy and value function in the PPO algorithm

    :param type observation_shape: dimensions of obs space.
    :param type sub_indices: the indices of actions in contact and non-contact
        sub spaces.
    :param type hidden_sizes: the number of nodes and layers for hidden layers.

    """
    def __init__(
            self,
            observation_shape,
            n_actions,
            sub_indices,
            hidden_sizes=None,  # None for default (see below).
            hidden_nonlinearity=torch.nn.Tanh,  # Module form.
    ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self.input_size = int(np.prod(observation_shape))
        hidden_sizes = hidden_sizes or DEFAULT_HIDDEN_LAYERS
        self.sub_indices = sub_indices
        self.n = n_actions
        # TODO: can we share some layers between 3 networks here?
        self.pi_free = MlpModel(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            output_size=len(sub_indices[0]),
            nonlinearity=hidden_nonlinearity,
        )
        self.pi_con = MlpModel(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            output_size=len(sub_indices[1]),
            nonlinearity=hidden_nonlinearity,
        )

        self.v = MlpModel(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            nonlinearity=hidden_nonlinearity,
        )

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute mean, log_std, and value estimate from input state. Infers
        leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Intermediate feedforward layers
        process as [T*B,H], with T=1,B=1 when not given. Used both in sampler
        and in algorithm (both via the agent).
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        obs_flat = observation.view(T * B, -1)
        pi = torch.zeros((obs_flat.shape[0], self.n),
                         dtype=obs_flat.dtype,
                         device=obs_flat.device)

        for i in range(obs_flat.shape[0]):
            if (torch.abs(obs_flat[i, self.input_size:self.input_size + 3]) <=
                    5e-1).all():
                p = F.softmax(self.pi_free(obs_flat[i, :self.input_size]),
                              dim=-1)
                pi[i, self.sub_indices[0]] = p
            else:
                p = F.softmax(self.pi_con(obs_flat[i, :self.input_size]),
                              dim=-1)
                pi[i, self.sub_indices[1]] = p

        v = self.v(obs_flat[:, :self.input_size]).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v


class PPOStructuredRealModel(PPOStructuredInsertionModel):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute mean, log_std, and value estimate from input state. Infers
        leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Intermediate feedforward layers
        process as [T*B,H], with T=1,B=1 when not given. Used both in sampler
        and in algorithm (both via the agent).
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        obs_flat = observation.view(T * B, -1)
        pi = torch.zeros((obs_flat.shape[0], self.n),
                         dtype=obs_flat.dtype,
                         device=obs_flat.device)

        for i in range(obs_flat.shape[0]):
            if (np.abs(obs_flat[i, self.input_size:self.input_size + 3]) <=
                    2).all():
                p = F.softmax(self.pi_free(obs_flat[i, :self.input_size]),
                              dim=-1)
                pi[i, self.sub_indices[0]] = p
            else:
                p = F.softmax(self.pi_con(obs_flat[i, :self.input_size]),
                              dim=-1)
                pi[i, self.sub_indices[1]] = p

        v = self.v(obs_flat[:, :self.input_size]).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v


class PPOStructuredInsertionAgent(StructuredInsertionEnvMixin, CustomAgent):
    def __init__(self, ModelCls=PPOStructuredInsertionModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class PPOStructuredRealAgent(StructuredInsertionEnvMixin, CustomAgent):
    def __init__(self, ModelCls=PPOStructuredRealModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
