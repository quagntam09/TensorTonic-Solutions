import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    real = np.asarray(real_scores, dtype = np.float32).flatten()
    fake = np.asarray(fake_scores, dtype = np.float32).flatten()

    loss = np.mean(fake) - np.mean(real)

    return float(loss)