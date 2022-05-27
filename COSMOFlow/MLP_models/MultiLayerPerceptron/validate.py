import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from MultiLayerPerceptron.utilities import norm_inputs, unnorm, get_script_path
import seaborn as sns
sns.set_theme()


def run_on_dataset(model, test_data,label_dim =None,  distances=None, n_batches=1, device=None, y_transform_fn=None, runtime=False):
    """
    Get the re-processed output of the supplied model on a set of supplied test data.

    Args:
        model (LinearModel): Model to test on `test_data`
        test_data (2-tuple/list): Tuple or list of the 'features' and their corresponding 'labels'.
        distances (ndarray): List of luminosity distance measurements for the input events. If None, results will not be scaled by luminosity distance. Note that this scaling is applied after the data is converted with y_transform_fn.
        n_batches (int, optional): Number of batches to process the input data in. Defaults to 1 (the entire dataset).
        device (string, optional): Device to attach the input model to, if it is not attached already.
        y_transform_fn (function, optional): If the labels/ydata have been pre-processed with a function (e.g. log),
                                             this function must be supplied to undo this if comparison is to be made
                                             in the unaltered function space.
        runtime (bool, optional): If True, return timing statistics for the model on the provided dataset.
                                  Defaults to False.

    Returns:
        tuple containing:
            output_data (double): Neural network output for this dataset.
            total_time (double): total time taken by the model to process the input dataset.
                                 Only if `runtime` is set to True.
            per_point (double): mean time taken by the model per sample in the input dataset.
                                 Only if `runtime` is set to True.
    """
    if device is not None:
        model = model.to(device)
    model.eval()

    xdata = test_data
    ydata_size = xdata[:,0].size
    xscalevals = np.load(get_script_path() + f'/models/{model.name}/xdata_inputs.npy')
    yscalevals = np.load(get_script_path() + f'/models/{model.name}/ydata_inputs.npy')

    test_input = torch.Tensor(xdata)
    normed_input = norm_inputs(test_input, ref_inputs=xscalevals,norm_type=model.norm_type).float().to(device)

    if runtime:
        st = time.perf_counter()
    with torch.no_grad():
        out = []
        for i in range(n_batches):
            output = model(normed_input[i * ydata_size // n_batches: (i + 1) * ydata_size// n_batches])
            out.append(output.detach().cpu().numpy())

    if runtime:
        et = time.perf_counter()
        total_time = et - st
        per_point = (et - st) / ydata_size

    output = np.concatenate(out)
    out_unnorm = unnorm(output, ref_inputs=yscalevals,norm_type=model.norm_type)

    if y_transform_fn is not None:
        out_unnorm = y_transform_fn(out_unnorm)
    
    if distances is None:
        distances = np.ones(xdata.shape[0]) * 0.5
    
    out_unnorm *= (0.5/distances)[:,None]
    
    if xdata[:,0].ndim == 1:
        out_unnorm = out_unnorm.flatten()
    outputs = (out_unnorm,)

    if runtime:
        outputs += (total_time, per_point,)

    return outputs


def compute_rmse(comparison_sets):
    """
    Get the root-mean-square-error difference between two sets of ordered data.
    Args:
        comparison_sets (2-tuple): Tuple of the list of true values, followed by the list of predicted values.
    Returns:
        rmse (double): root-mean-square-error between the truth and the model output for this dataset.
    """
    truth, pred = comparison_sets
    rmse = np.sqrt(np.mean((truth.flatten() - pred.flatten()) ** 2))
    return rmse


def test_threshold_accuracy(comparison_sets, threshold, confusion_matrix=False):
    truth, pred = comparison_sets
    out_classified = np.zeros(shape=pred.size)
    out_classified[pred.flatten() >= threshold] = 1

    truth_classified = np.zeros(shape=truth.size)
    truth_classified[truth.flatten() >= threshold] = 1

    if not confusion_matrix:
        return 1 - np.mean(np.abs(out_classified - truth_classified))
    else:
        confmat = np.zeros((2,2))
        confmat[0,0] = np.sum(np.logical_and(out_classified==0,truth_classified==0))
        confmat[0,1] = np.sum(np.logical_and(out_classified==0,truth_classified==1))
        confmat[1,0] = np.sum(np.logical_and(out_classified==1,truth_classified==0))
        confmat[1,1] = np.sum(np.logical_and(out_classified==1,truth_classified==1))

        return (1-np.mean(np.abs(out_classified-truth_classified)),confmat)

def plot_histograms(comparison_sets, model_name, xlabel, title=None, title_kwargs={}, xlabel_kwargs={}, log=True,
                    fig_kwargs={}, plot_kwargs={}, save_kwargs={}, legend_kwargs={}):
    truth, pred = comparison_sets
    if log:
        truth = np.log10(truth)
        pred = np.log10(pred)

    fig, ax = plt.subplots(**fig_kwargs)

    try:
        bins = plot_kwargs.pop('bins')
    except KeyError:
        bins = 'auto'

    ax.hist(truth, bins=bins, label='Truth', **plot_kwargs)
    ax.hist(pred, bins=bins, label='Prediction', **plot_kwargs)

    ax.set_xlabel(xlabel, **xlabel_kwargs)

    if title is not None:
        ax.set_title(title, **title_kwargs)

    ax.legend(**legend_kwargs)
    plt.savefig(get_script_path() + f'/models/{model_name}/hists.png', **save_kwargs)
    plt.close()


def plot_difference_histogram(comparison_sets, model_name, xlabel, title=None, title_kwargs={}, xlabel_kwargs={},
                              log=True, one_sided=False, ratio=False, fig_kwargs={}, plot_kwargs={}, save_kwargs={}):
    truth, pred = comparison_sets
    if log:
        truth = np.log10(truth)
        pred = np.log10(pred)

    fig, ax = plt.subplots(**fig_kwargs)

    try:
        bins = plot_kwargs.pop('bins')
    except KeyError:
        bins = 'auto'

    if ratio:
        out = pred / truth
        if log:
            out = np.log10(out)
            if one_sided:
                out = abs(out)
    else:
        out = pred - truth
        if log:
            out = np.log10(abs(out))

    ax.hist(out, bins=bins, **plot_kwargs)

    ax.set_xlabel(xlabel, **xlabel_kwargs)

    if title is not None:
        ax.set_title(title, **title_kwargs)

    if ratio:
        diff = 'rel'
    else:
        diff = 'abs'
    if log:
        logname = 'log'
    else:
        logname = ''

    plt.savefig(get_script_path() + f'/models/{model_name}/hist_{logname}{diff}diff.png', **save_kwargs)
    plt.close()



