import pickle
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from sklearn.metrics import f1_score



def batch_to_jax(batch):
    return {k: jnp.array(v) for k, v in batch.items()}

def evaluate_f1_score(reward_model, batch):
    logits = reward_model._eval_pref_step(reward_model.train_states, jax.random.PRNGKey(0), batch)
    threshold = 0.5
    pred_labels = (logits[:, 0] >= threshold).astype(int)
    true_labels = batch['labels'][0].astype(int)
    true_label= np.full(pred_labels.shape, true_labels[0])
    f1 = f1_score(true_label, pred_labels)
    return f1


if __name__ == "__main__":
    wandb.init(project="chess_pt", name="F1_Score_by_Rating",entity='hails' , config={
        "num_eval_queries": 200,
        "query_len": 50,
    })

    model_path = "/home/hail/PreferenceTransformer/JaxPref/reward_model/chess/PrefTransformer/Classical_expert_low/s7/model.pkl"
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    reward_model = data['reward_model']
    f1_scores = []
    num_eval_queries = 200
    query_len = 50

    for rating in range(1300, 1701, 100):
        eval_path = f'/home/hail/Chess_data/Classical/convert_classical_medium_low_f1/combined_C_eval_batch_{rating:02d}.pkl'
        with open(eval_path, "rb") as f:
            eval_batch = pickle.load(f)
        jax_eval_batch = batch_to_jax(eval_batch)
        f1_score_value = evaluate_f1_score(reward_model, jax_eval_batch)
        f1_scores.append(f1_score_value)
        wandb.log({"Rating": rating, "F1 Score_CEL": f1_score_value})





