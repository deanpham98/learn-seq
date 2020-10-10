import numpy as np
import torch
import torch.nn.functional as F
from rlpyt.models.mlp import MlpModel
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

DEFAULT_HIDDEN_LAYERS = [64, 64]

# mixin is to convert env specs to model parameters
class InsertionEnvMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)

class StructuredInsertionEnvMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        sub_spaces = env_spaces.action.space.sub_spaces
        sub_indices = env_spaces.action.space.sub_indices
        output_size = [space.n for space in sub_spaces]
        n = env_spaces.action.n
        # print(output_size)
        return dict(observation_shape=env_spaces.observation.shape,
                    n_actions=n,
                    sub_indices=sub_indices)

class PPOStructuredInsertionModel(torch.nn.Module):
    """Implement the policy and value function in the PPO algorithm

    :param type observation_shape: Description of parameter `observation_shape`.
    :param type sub_indices: Description of parameter `sub_indices`.
    :param type hidden_sizes: Description of parameter `hidden_sizes`.

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
        pi = torch.zeros((obs_flat.shape[0], self.n), dtype=obs_flat.dtype, device=obs_flat.device)

        for i in range(obs_flat.shape[0]):
            if (torch.abs(obs_flat[i, self.input_size:self.input_size+3])<=1e-1).all():
                p = F.softmax(self.pi_free(obs_flat[i, :self.input_size]), dim=-1)
                pi[i, self.sub_indices[0]] = p
            else:
                p = F.softmax(self.pi_con(obs_flat[i, :self.input_size]), dim=-1)
                pi[i, self.sub_indices[1]] = p

        v = self.v(obs_flat[:, :self.input_size]).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v

class PPOStructuredInsertionAgent(StructuredInsertionEnvMixin, CategoricalPgAgent):
    def __init__(self, ModelCls=PPOStructuredInsertionModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
