from handle_flow_new_v1 import HandleFlow
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import corner
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import bilby
from tqdm import tqdm
from bilby.core.prior import Uniform
from scipy.stats import entropy
import argparse
bilby.core.utils.log.setup_logger(log_level=0)

# Argument parser to define Nruns and flow_name
parser = argparse.ArgumentParser(description='Run PP plot and H0 analysis')
parser.add_argument('--Nresults', type=int, default=1, help='Number of results for PP plots')
parser.add_argument('--flow_name', type=str, default='Flow_Testing_O3_H1_L1_V1_14target_1Cond', help='Name of the flow to use')
args = parser.parse_args()
Nruns = 1
Nresults  = args.Nresults
flow_name = args.flow_name
path = 'trained_flows_and_curves/'+flow_name+'/'
# Usage example:
n = 10000
print('Loading training data...')
flow_handler = HandleFlow(path='trained_flows_and_curves/', flow_name=flow_name, device='cpu')
samples = flow_handler.sample_flow(conditional=np.random.uniform(20, 140, n), n_samples=n)

# Load training data
training_data = flow_handler.hyperparameters['training_data_path']
GWdata = pd.read_csv(training_data.format(1), skipinitialspace=True)
dataGW = GWdata[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_jl', 'phi_12', 'psi', 'geocent_time']]

# Function to calculate KL divergence
def kl_divergence(p, q, start=-np.inf, end=np.inf):
    return quad(lambda x: p(x) * np.log(p(x) / q(x)), start, end)[0]

# Function to calculate JS divergence
def js_divergence(p, q, start, end):
    kl_div_pq = kl_divergence(p, q, start, end)
    kl_div_qp = kl_divergence(q, p, start, end)
    return 0.5 * (kl_div_pq + kl_div_qp)

# Assuming samples and dataGW are available
# Dictionary of random samples
dict_rand = {'luminosity_distance': list(samples[:, 0]), 'ra': list(samples[:, 1]), 'dec': list(samples[:, 2]),
             'mass_1': list(samples[:, 3]), 'mass_2': list(samples[:, 4]),
             'a_1': list(samples[:, 5]), 'a_2': list(samples[:, 6]), 'tilt_1': list(samples[:, 7]),
             'tilt_2': list(samples[:, 8]), 'theta_jn': list(samples[:, 9]), 'phi_jl': list(samples[:, 10]),
             'phi_12': list(samples[:, 11]), 'psi': list(samples[:, 12]), 'geocent_time': list(samples[:, 13])}

JS = []
for key in dict_rand.keys():
    # Generate sample data for the given dimension
    data = dataGW[[key]].to_numpy().flatten()
    samples_dim = np.array(dict_rand[key])

    # Estimate the PDFs using KDE
    pdf_p = gaussian_kde(data[:100])
    pdf_q = gaussian_kde(samples_dim[:100])

    # Calculate JS divergence
    js = js_divergence(pdf_p.evaluate, pdf_q.evaluate, start=data.min(), end=data.max())
    JS.append(js)
    print(f"JS divergence for {key}: {js}")

# Create DataFrame for the flow samples
samples_flow = pd.DataFrame(dict_rand)

# Plotting corner plots for flow samples and target data
c1 = corner.corner(samples_flow, bins=30, plot_density=False, plot_datapoints=False, smooth=True,
                   levels=(0.5, 0.99), color='red', hist_kwargs={'density': 1, 'linewidth': 3},
                   range=[(0, 6000), (0, 2 * np.pi), (-np.pi / 2, np.pi / 2), (0, 80), (0, 80),
                          (0, 0.999), (0, 0.999), (0, np.pi), (0, np.pi), (0, np.pi), (0, 2 * np.pi),
                          (0, 2 * np.pi), (0, np.pi), (0, 86164)])

# Plotting target data
data_plot = dataGW[['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2',
                    'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_jl',
                    'phi_12', 'psi', 'geocent_time']].sample(n=n)

fig = corner.corner(data_plot, bins=30,
                    plot_datapoints=False,
                    plot_density=False,
                    smooth=True,
                    fig=c1,
                    levels=(0.5, 0.99),
                    labels=[r'$D_{L}[Mpc]$', r'$\alpha[rad]$', r'$\delta[rad]$',
                            r'$m_{1,z}[M_{\odot}]$', r'$m_{2,z}[M_{\odot}]$', r'$a_1$', r'$a_2$', r'$\theta_1[rad]$', r'$\theta_2[rad]$',
                            r'$\theta_{JN}[rad]$', r'$\phi_{JL}[rad]$', r'$\phi_{12}[rad]$', r'$\psi[rad]$', r'$t_{geo}[s]$'],
                    label_kwargs={'fontsize': 25},
                    hist_kwargs={'density': 1, 'linewidth': 3},
                    range=[(0, 6000), (0, 2 * np.pi), (-np.pi / 2, np.pi / 2), (0, 80), (0, 80),
                           (0, 0.999), (0, 0.999), (0, np.pi), (0, np.pi), (0, np.pi), (0, 2 * np.pi),
                           (0, 2 * np.pi), (0, np.pi), (0, 86164)])

# Create proxy artists for the legend
blue_line = plt.Line2D([0], [0], linewidth=10, color='black', label='Target Data')
red_line = plt.Line2D([0], [0], linewidth=10, color='red', label='Normalising Flow')

# Adding JS divergence information to the legend
legend_lines = [blue_line, red_line]
labels = [r'$D_{L} \:\:$', r'$RA \:\:$', r'$\delta \:\:$', r'$m_{1} \:\:$', r'$m_{2} \:\:$',
          r'$a_{1} \:\:$', r'$a_{2} \:\:$', r'$\theta_{1} \:\:$', r'$\theta_{2} \:\:$',
          r'$\theta_{JN} \:\:$', r'$\phi_{JL} \:\:$', r'$\phi_{12} \:\:$', r'$\psi \:\:$', r'$t_{geo} \:\:$']

for i, label in enumerate(labels):
    legend_lines.append(plt.Line2D([0], [0], linestyle='none', label=label + f"JS= {JS[i] * 1000:.1f} millinats"))

# Add legend to the figure, outside the corner plots
fig.legend(handles=legend_lines, loc=[0.55, 0.65], frameon=True, fontsize=31)
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.savefig(path+'/test_resampling.png', dpi = 500)

# Display the plot
plt.show()

# PP plot for GW parameters from resampling
print('Making PP plot for GW parameters from flow...')
N = 50000
np.random.seed(4321)
labels = ['luminosity_distance', 'ra', 'dec', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_jl', 'phi_12', 'psi', 'geocent_time']
label_keys = [r'$D_{L}$', r'$\alpha$', r'$\delta$', r'$m_{1,z} \: [M_{\odot}]$', r'$m_{2,z} \: [M_{\odot}]$', r'$a_1$', r'$a_2$', r'$	heta_1$', r'$	heta_2$', r'$	heta_{JN}$', r'$\phi_{JL}$', r'$\phi_{12}$', r'$\psi$', r'$t_{geo}$']
true_samples = pd.DataFrame(dict_rand)
priors = {f"{labels[jj]}": Uniform(0, 1, f"{labels[jj]}") for jj in range(14)}

for x in range(Nruns):
    results = []
    for ii in tqdm(range(Nresults)):
        posterior = dict()
        injections = dict()
        indicies = np.arange(len(np.array(GWdata[['H0']])))
        inx = np.random.choice(indicies, size=1, replace=False)
        i = 0
        conditional_sample = np.array(GWdata[['H0']])[inx]
        conditional_sample = conditional_sample.reshape(-1, 1) * np.ones((len(conditional_sample), N))
        samples = flow_handler.sample_flow(conditional=conditional_sample, n_samples=N)
        for key, prior in priors.items():
            truths = np.array(GWdata[key])[inx]
            posterior[key] = samples[:, i]
            injections[key] = truths.astype('float32').item()
            i += 1

        posterior = pd.DataFrame(dict(posterior))
        result = bilby.result.Result(
            injection_parameters=injections,
            posterior=posterior,
            search_parameter_keys=injections.keys(),
            priors=priors)
        results.append(result)

    fig = bilby.result.make_pp_plot(results, filename=path+'PP_GW_para',
                                    confidence_interval=(0.68, 0.90, 0.99, 0.9999))

# Customize the plot
fig[0].set_size_inches(12, 10)  # Increase figure size
ax = fig[0].gca()  # Get the current axes of the figure
ideal_line, = ax.plot([0, 1], [0, 1], 'k--', label='Ideal')  # Add a black dashed line at 45 degrees
ax.set_title('')
ax.set_xlabel('Confidence interval (C.I.)', fontsize=20)
ax.set_ylabel('Fraction of events in C.I.', fontsize=15)
p_values = fig[1][1]
# Append p-values to label_keys
label_keys_with_p = [f"{label_keys[i]} (p={p_values[i]:.2f})" for i in range(len(label_keys))]

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=14)

# Customize the legend with label_keys and include the ideal line
handles, labels = ax.get_legend_handles_labels()
handles.append(ideal_line)
label_keys_with_p.append('Ideal')
ax.legend(handles, label_keys_with_p, fontsize=12, ncol=3)

# If you need to save the figure again after adding the line
plt.savefig(path + 'PP_GW_para.png', dpi=300)
plt.show()

# Histograms of H0 samples with true values
# Constants and parameters
Nomega = 100
h0 = np.linspace(20, 140, Nomega)

# Extract data
h0s = GWdata[['H0']]
print('Making H0 histograms for PP plot...')
# Prepare the plot
fig, axs = plt.subplots(4, 3, figsize=(12, 12))

# Generate subplots
for idx, ax in enumerate(axs.flatten()):
    if idx >= 12:
        break

    likelihood_vertical = []
    inx_sample = np.random.randint(len(GWdata))
    log_like = np.zeros(Nomega)
    gw_sample = dataGW.iloc[inx_sample, :]  # Extracting a sample

    if isinstance(gw_sample, pd.Series):
        gw_sample = gw_sample.to_frame().T  # Convert Series to DataFrame

    for i, h in enumerate(h0):
        log_like[i] = flow_handler.evaluate_log_prob(gw_sample, h)[0]
    like = np.exp(log_like)
    dh = np.diff(h0)[0]
    like /= np.trapz(like, x=h0)
    cdf = np.cumsum(like * np.diff(h0, prepend=h0[0]))
    cdf /= cdf[-1]  # Normalize CDF to range from 0 to 1

    t = np.random.uniform(0, 1, size=n)
    samples_h0 = np.interp(t, cdf, h0)
    ax.hist(samples_h0, density=True, bins=50, linewidth=3, label=r'samples')
    ax.plot(h0, like / np.trapz(like, x=h0), label=r'$p(H_{0}|\theta, D)$', linewidth=2)

    ax.axvline(x=np.array(h0s.iloc[inx_sample]), color='r', linestyle='-', linewidth=2, label='True $H_0$')

    ax.set_ylim([0, 0.02])
    ax.set_xlim([20, 140])
    if idx % 3 == 0:
        ax.set_ylabel('Density', fontsize=12)
    if idx > 8:
        ax.set_xlabel('$H_0 \: [km \: s^{-1} \: Mpc^{-1}]$', fontsize=13)
    if idx == 0:
        ax.legend()

plt.tight_layout()
plt.savefig(path+'/H0_resampling.png', dpi = 500)
plt.show()

# PP plot for H0
priors = {'H0': Uniform(0, 1, 'H0')}

# Function to calculate likelihood and sample
def calculate_likelihood_and_samples(flow_class, GWdata, h0, Nomega, inx_sample):
    like = np.zeros(Nomega)
    gw_sample = GWdata.iloc[inx_sample, :]

    if isinstance(gw_sample, pd.Series):
        gw_sample = gw_sample.to_frame().T

    for i, h in enumerate(h0):
        log_like[i] = flow_handler.evaluate_log_prob(gw_sample, h)[0]

    like = np.exp(log_like)
    # Normalize and calculate CDF
    like /= np.trapz(like, x=h0)
    cdf = np.cumsum(like * np.diff(h0, prepend=h0[0]))
    cdf /= cdf[-1]

    # Inverse transform sampling
    t = np.random.uniform(0, 1, size=n)
    samples = np.interp(t, cdf, h0)

    return samples

# Function to create PP plot
def create_pp_plot(flow_class, GWdata, h0s, priors, Nresults=100, Nruns=1):
    priors = {label: priors.get(label, bilby.core.prior.Uniform(0, 1, label)) for label in h0s.columns}
    results = []

    for _ in range(Nruns):
        for _ in tqdm(range(Nresults)):
            posterior = {}
            injections = {}
            inx = np.random.randint(len(GWdata))
            samples = calculate_likelihood_and_samples(flow_class, GWdata, h0, Nomega, inx)

            for i, key in enumerate(h0s.columns):
                posterior[key] = samples
                injections[key] = h0s.iloc[inx]['H0']

            result = bilby.result.Result(
                injection_parameters=injections,
                posterior=pd.DataFrame(posterior),
                search_parameter_keys=list(injections.keys()),
                priors=priors
            )
            results.append(result)

    fig = bilby.result.make_pp_plot(results, filename=path+ 'PP_H0', confidence_interval=(0.68, 0.90, 0.99, 0.9999))
    fig[0].set_size_inches(12, 10)
    ax = fig[0].gca()
    ideal_line, = ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
    ax.set_xlabel('Confidence interval (C.I.)', fontsize=20)
    ax.set_ylabel('Fraction of events in C.I.', fontsize=15)

    p_values = fig[1][1]
    label_keys = [f"{label} (p={p_values[i]:.2f})" for i, label in enumerate(h0s.columns) if i < len(p_values)]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([ideal_line] + handles, ['Ideal'] + label_keys, fontsize=12, ncol=3)
    plt.show()

print('Making PP H0 plot...')
# Create PP plot
create_pp_plot(flow_handler, dataGW, h0s, priors, Nresults=Nresults, Nruns=1)