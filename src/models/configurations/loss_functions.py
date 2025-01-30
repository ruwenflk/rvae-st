import tensorflow as tf


def _reduce_mean(tensor, axis=None):
    return tf.math.reduce_mean(tensor, axis=axis)


def _reduce_sum(tensor, axis=None):
    return tf.math.reduce_sum(tensor, axis=axis)


def _exp(tensor):
    return tf.exp(tensor)


def _square(tensor):
    return tf.square(tensor)


def _reshape(tensor, shape):
    return tf.reshape(tensor=tensor, shape=shape)


def _square(tensor):
    return tf.square(tensor)


def get_kl_loss(mu, log_var):
    kl_divergence_loss = 1 + log_var - _square(mu) - _exp(log_var)
    return -0.5 * _reduce_sum(
        kl_divergence_loss, axis=1
    )  # tf.math.reduce_sum(kl_divergence_loss, axis=1)


def get_squared_error(data, data_recon):
    batchsize = data.shape[0]
    deltas = _reshape(data, (batchsize, -1)) - _reshape(data_recon, (batchsize, -1))
    recon_loss = _square(deltas)
    return recon_loss


def get_mse(data, data_recon):
    squared_error = get_squared_error(data, data_recon)
    return _reduce_sum(
        squared_error, axis=1
    )  # tf.math.reduce_sum(squared_error, axis=1)


def get_nll_loss(y, mu_hat, logvar_hat):
    logvar_hat = _reshape(tensor=logvar_hat, shape=(y.shape[0], -1))
    var_hat = _exp(logvar_hat)

    MSE = get_squared_error(y, mu_hat)
    loss = MSE / var_hat + logvar_hat

    return 0.5 * _reduce_sum(loss, axis=1)  # tf.math.reduce_sum(loss, axis=1)


def mse_kl(data, data_recon, mu, log_var, alpha, beta):
    kl_loss = get_kl_loss(mu, log_var)
    kl_loss_avg = _reduce_mean(kl_loss)

    recon_loss = get_mse(data, data_recon)
    recon_loss_avg = _reduce_mean(recon_loss)

    final_loss = (alpha * recon_loss_avg) + (beta * kl_loss_avg)

    return final_loss, recon_loss_avg, kl_loss_avg


def mse_kl_no_batch_avg(data, data_recon, mu, log_var, alpha, beta):
    kl_loss = get_kl_loss(mu, log_var)
    # kl_loss_avg = _reduce_mean(kl_loss)

    recon_loss = get_mse(data, data_recon)
    # recon_loss_avg = _reduce_mean(recon_loss)

    final_loss = (alpha * recon_loss) + (beta * kl_loss)

    return final_loss, recon_loss, kl_loss


def timevaeloss(data, data_recon, mu, log_var, alpha, beta):
    def get_reconst_loss_by_axis(X, X_c, axis):
        x_r = tf.reduce_mean(X, axis=axis)
        x_c_r = tf.reduce_mean(X_c, axis=axis)
        err = tf.math.squared_difference(x_r, x_c_r)
        loss = tf.reduce_sum(err)
        return loss

    # overall
    err = tf.math.squared_difference(data, data_recon)
    recon_loss_avg = tf.reduce_sum(err)

    recon_loss_avg += get_reconst_loss_by_axis(
        data, data_recon, axis=[2]
    )  # by time axis
    # reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])    # by feature axis

    kl_loss = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
    kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
    final_loss = (alpha * recon_loss_avg) + (beta * kl_loss)

    return final_loss, recon_loss_avg, kl_loss


def nll_kl(y, mu_hat, mu, logvar_hat, logvar, beta):
    kl_loss = get_kl_loss(mu, logvar)
    kl_loss_avg = _reduce_mean(kl_loss)

    mu_var_loss = get_nll_loss(y, mu_hat, logvar_hat)
    mu_var_loss_avg = _reduce_mean(mu_var_loss)

    final_loss = mu_var_loss_avg + (beta * kl_loss_avg)
    return final_loss, mu_var_loss_avg, kl_loss_avg


# remove when no old config needs the recon_kl anymore
def recon_kl(data, data_recon, mu, log_var, alpha, beta):
    return mse_kl(data, data_recon, mu, log_var, alpha, beta)
