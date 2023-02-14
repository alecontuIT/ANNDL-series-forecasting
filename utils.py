import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
tfk = tf.keras

def build_sequences(df, target_labels, window=200, stride=20, telescope=100):
    # Sanity check to avoid runtime errors
    assert window % stride == 0
    dataset = []
    labels = []
    temp_df = df.copy().values
    temp_label = df[target_labels].copy().values
    padding_len = len(df)%window

    if(padding_len != 0):
        # Compute padding length
        padding_len = window - len(df)%window
        padding = np.zeros((padding_len,temp_df.shape[1]), dtype='float64')
        temp_df = np.concatenate((padding,df))
        padding = np.zeros((padding_len,temp_label.shape[1]), dtype='float64')
        temp_label = np.concatenate((padding,temp_label))
        assert len(temp_df) % window == 0

    for idx in np.arange(0,len(temp_df)-window-telescope,stride):
        dataset.append(temp_df[idx:idx+window])
        labels.append(temp_label[idx+window:idx+window+telescope])

    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels


def inspect_dataframe(df, columns):
    figs, axs = plt.subplots(len(columns), 1, sharex=True, figsize=(17,17))
    for i, col in enumerate(columns):
        axs[i].plot(df[col])
        axs[i].set_title(col)
    plt.show()

def inspect_multivariate(x, y, columns, telescope, idx=None):
    if(idx==None):
        idx=np.random.randint(0,len(x))

    figs, axs = plt.subplots(len(columns), 1, sharex=True, figsize=(17,17))
    for i, col in enumerate(columns):
        axs[i].plot(np.arange(len(x[0,:,i])), x[idx,:,i])
        axs[i].scatter(np.arange(len(x[0,:,i]), len(x[0,:,i]) + telescope), y[idx,:,i], color='orange')
        axs[i].set_title(col)
        axs[i].set_ylim(0,1)
    plt.show()

def callbacks(which_monitor, maxOrMin, patience, model_version, models_folder, factor=0.5, min_lr=1e-5):
    """
    which_monitor -> string to identify monitor metric in EarlyStopping
    maxOrMin      -> string to identify if in EarlyStopping we have to look at the max or at the min of
    the monitor before stopping
    patience      -> how much we have to wait, once it's reached the max or min to stop

    return        -> an array of callbacks (Early stopping and checkpoints of the best and of the last)
    """
    callbacks = []

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    path_to_model_version = os.path.join(models_folder, model_version)
    if not os.path.exists(path_to_model_version):
        os.makedirs(path_to_model_version)

    # Model checkpoint -> automatically save the model during training
    ckpt_dir = os.path.join(path_to_model_version, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_last.ckpt.h5'),
                                                     save_weights_only=False,
                                                     save_best_only=False)
    callbacks.append(ckpt_callback)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_best.ckpt.h5'),
                                                     save_weights_only=False,
                                                     save_best_only=True)

    callbacks.append(ckpt_callback)

    # Visualize Learning on Tensorboard
    tb_dir = os.path.join(path_to_model_version, 'tb_logs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # By default shows losses and metrics for both training and validation
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                               profile_batch=0,
                                               histogram_freq=1)
    callbacks.append(tb_callback)

    # Early Stopping
    es_callback = tf.keras.callbacks.EarlyStopping(monitor=which_monitor, mode=maxOrMin,
                                                   patience=patience, restore_best_weights=True)
    callbacks.append(es_callback)

    rLROnPlateau = tfk.callbacks.ReduceLROnPlateau(monitor=which_monitor, mode=maxOrMin, patience=(patience/2), factor=factor, min_lr=min_lr)
    callbacks.append(rLROnPlateau)

    return callbacks



def metrics():
    '''
    returns an array of metrics usefull to evaluate a model during and after training
    '''
    _metrics = [
      tfk.metrics.MeanAbsoluteError(name='mae'),
      tfk.metrics.MeanAbsolutePercentageError(name='mape'),
      tfk.metrics.MeanSquaredError(name='mse'),
      tfk.metrics.MeanSquaredLogarithmicError(name='msle'),
      tfk.metrics.RootMeanSquaredError(name='rmse'),
    ]
    return _metrics

def plot_and_save(history, model_version, models_folder):
    # this is to save history on file
    path_to_model = os.path.join(os.path.join('.', models_folder), model_version)
    np.save(os.path.join(path_to_model, 'history.npy'), history)

    # plot of history metrics
    path_to_plots = os.path.join(os.path.join(os.path.join('.', models_folder), model_version), 'metrics_plots')
    keys = list(history.keys())
    number_of_metrics = len(keys)
    for idx in range(0, int((number_of_metrics - 1) / 2)):
        plt.figure(figsize=(15,5))
        plt.plot(history[keys[idx]], label=keys[idx].upper()+' Training', alpha=.8, color='#ff00ff')
        plt.plot(history[keys[idx + int((number_of_metrics - 1) / 2)]], label=keys[idx].upper()+' Validation', alpha=.8, color='#00ffff')
        plt.legend(loc='upper left')
        plt.title(keys[idx].upper())
        plt.grid(alpha=.3)

        if not os.path.exists(path_to_plots):
            os.makedirs(path_to_plots)
        img_name = os.path.join(path_to_plots, keys[idx] + '_plot.png')
        plt.savefig(img_name)

    # To plot Learning rate
    plt.figure(figsize=(15,5))
    plt.plot(history[keys[number_of_metrics-1]], label='Learning Rate', alpha=.8, color='#00ff11')
    plt.legend(loc='upper left')
    plt.title(str('Learning Rate').upper())
    plt.grid(alpha=.3)
    plt.savefig(os.path.join(path_to_plots, 'lr_plot.png'))

def inspect_multivariate_prediction(X, pred, columns, telescope, models_folder, model_version, idx=None):
    path_to_plots = os.path.join(os.path.join(os.path.join('.', models_folder), model_version), 'predictions_plots')

    for i, col in enumerate(columns):
        plt.figure(figsize=(15,5))
        plt.plot(np.arange(len(X[-3000:,i])), X[-3000:,i], label = 'Past')
        plt.plot(np.arange(len(X[-3000:,i]), len(X[-3000:,i])+telescope), pred[:,i], label='Predictions', color='green')
        plt.title(col)
        plt.legend(loc='upper left')
        if not os.path.exists(path_to_plots):
            os.makedirs(path_to_plots)
        img_name = os.path.join(path_to_plots, col + '_plot.png')
        plt.savefig(img_name)
    plt.show()

def predict_with_AR(model, preprocessed_past_data, telescope):
    out = np.array([])
    x_tmp = preprocessed_past_data
    for reg in range(telescope):
        pred_tmp = model.predict(x_tmp)
        if(len(out)==0):
            out = pred_tmp
        else:
            out = np.concatenate((out,pred_tmp),axis=1)
        x_tmp = np.concatenate((x_tmp[:,1:,:],pred_tmp), axis=1)
    return out

def preprocessing(dataset, window):
    x = dataset.to_numpy()
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)

    past_data = x[-window:]
    past_data_prep = (past_data - x_min) / (x_max - x_min)
    past_data_prep = np.expand_dims(past_data_prep, axis=0)
    return past_data_prep, x_min, x_max

def postprocessing(normalized_out, x_min, x_max):
    out = normalized_out * (x_max - x_min) + x_min
    out = np.reshape(out, (864, 7))
    return out

def processing_and_prediction(dataset, window, telescope, model):
    preprocessed_past_data, x_min, x_max = preprocessing(dataset, window)
    normalized_out = predict_with_AR(model, preprocessed_past_data, telescope)
    out = postprocessing(normalized_out, x_min, x_max)
    return out
