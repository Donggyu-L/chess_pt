import numpy as np
import os
import pickle

def initialize_model():
    model_path = os.path.join('/home/hail/PreferenceTransformer/JaxPref/reward_model/chess/PrefTransformer/Classical_expert_low/s7', "model.pkl")

    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt['reward_model']
    return reward_model


if __name__ == '__main__':
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    reward_model = initialize_model()

    state = np.random.randn(8, 8, 119)
    # state = state[np.newaxis, :]
    state_value = reward_model(state)
    print("State Value:", state_value)






def reward_from_preference_transformer(
        env_name: str,
        dataset: D4RLDataset,
        reward_model,
        seq_len: int,
        batch_size : int = 256,
        use_diff: bool = False,
        label_mode: str = 'last',
        with_attn_weights: bool = False # Option for attention analysis.
):
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations
    )
    trajectories = []
    trj_mapper = []
    observation_dim = dataset.observations.shape[-1]
    action_dim = dataset.actions.shape[-1]

    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="chunk trajectories"):
        _obs, _act, _reward, _mask, _done, _next_obs = [], [], [], [], [], []
        for _o, _a, _r, _m, _d, _no in traj:
            _obs.append(_o)
            _act.append(_a)
            _reward.append(_r)
            _mask.append(_m)
            _done.append(_d)
            _next_obs.append(_no)

        traj_len = len(traj)
        _obs, _act = np.asarray(_obs), np.asarray(_act)
        trajectories.append((_obs, _act))

        for seg_idx in range(traj_len):
            trj_mapper.append((trj_idx, seg_idx))

    data_size = dataset.rewards.shape[0]
    interval = int(data_size / batch_size) + 1
    new_r = np.zeros_like(dataset.rewards)
    pts = []
    attn_weights = []
    for i in trange(interval, desc="relabel reward"):
        start_pt = i * batch_size
        end_pt = min((i + 1) * batch_size, data_size)

        _input_obs, _input_act, _input_timestep, _input_attn_mask, _input_pt = [], [], [], [], []
        for pt in range(start_pt, end_pt):
            _trj_idx, _seg_idx = trj_mapper[pt]
            if _seg_idx < seq_len - 1:
                __input_obs = np.concatenate([np.zeros((seq_len - 1 - _seg_idx, observation_dim)), trajectories[_trj_idx][0][:_seg_idx + 1, :]], axis=0)
                __input_act = np.concatenate([np.zeros((seq_len - 1 - _seg_idx, action_dim)), trajectories[_trj_idx][1][:_seg_idx + 1, :]], axis=0)
                __input_timestep = np.concatenate([np.zeros(seq_len - 1 - _seg_idx, dtype=np.int32), np.arange(1, _seg_idx + 2, dtype=np.int32)], axis=0)
                __input_attn_mask = np.concatenate([np.zeros(seq_len - 1 - _seg_idx, dtype=np.int32), np.ones(_seg_idx + 1, dtype=np.float32)], axis=0)
                __input_pt = np.concatenate([np.zeros(seq_len - 1 - _seg_idx), np.arange(pt - _seg_idx , pt + 1)], axis=0)
            else:
                __input_obs = trajectories[_trj_idx][0][_seg_idx - seq_len + 1:_seg_idx + 1, :]
                __input_act = trajectories[_trj_idx][1][_seg_idx - seq_len + 1:_seg_idx + 1, :]
                __input_timestep = np.arange(1, seq_len + 1, dtype=np.int32)
                __input_attn_mask = np.ones((seq_len), dtype=np.float32)
                __input_pt = np.arange(pt - seq_len + 1, pt + 1)

            _input_obs.append(__input_obs)
            _input_act.append(__input_act)
            _input_timestep.append(__input_timestep)
            _input_attn_mask.append(__input_attn_mask)
            _input_pt.append(__input_pt)

        _input_obs = np.asarray(_input_obs)
        _input_act = np.asarray(_input_act)
        _input_timestep = np.asarray(_input_timestep)
        _input_attn_mask = np.asarray(_input_attn_mask)
        _input_pt = np.asarray(_input_pt)

        input = dict(
            observations=_input_obs,
            actions=_input_act,
            timestep=_input_timestep,
            attn_mask=_input_attn_mask,
        )

        jax_input = batch_to_jax(input)
        if with_attn_weights:
            new_reward, attn_weight = reward_model.get_reward(jax_input)
            attn_weights.append(np.array(attn_weight))
            pts.append(_input_pt)
        else:
            new_reward, _ = reward_model.get_reward(jax_input)
        new_reward = new_reward.reshape(end_pt - start_pt, seq_len) * _input_attn_mask

        if use_diff:
            prev_input = dict(
                observations=_input_obs[:, :seq_len - 1, :],
                actions=_input_act[:, :seq_len - 1, :],
                timestep=_input_timestep[:, :seq_len - 1],
                attn_mask=_input_attn_mask[:, :seq_len - 1],
            )
            jax_prev_input = batch_to_jax(prev_input)
            prev_reward, _ = reward_model.get_reward(jax_prev_input)
            prev_reward = prev_reward.reshape(end_pt - start_pt, seq_len - 1) * prev_input["attn_mask"]
            if label_mode == "mean":
                new_reward = jnp.sum(new_reward, axis=1).reshape(-1, 1)
                prev_reward = jnp.sum(prev_reward, axis=1).reshape(-1, 1)
            elif label_mode == "last":
                new_reward = new_reward[:, -1].reshape(-1, 1)
                prev_reward = prev_reward[:, -1].reshape(-1, 1)
            new_reward -= prev_reward
        else:
            if label_mode == "mean":
                new_reward = jnp.sum(new_reward, axis=1) / jnp.sum(_input_attn_mask, axis=1)
                new_reward = new_reward.reshape(-1, 1)
            elif label_mode == "last":
                new_reward = new_reward[:, -1].reshape(-1, 1)

        new_reward = np.asarray(list(new_reward))
        new_r[start_pt:end_pt, ...] = new_reward.squeeze(-1)

    dataset.rewards = new_r.copy()

    if with_attn_weights:
        return dataset, (attn_weights, pts)
    return dataset








