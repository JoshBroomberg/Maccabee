import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn'])

def display_data_metrics_comparison(
    metric_spec,
    concrete_data_agg, sampled_data_agg):

    for axes, metrics in metric_spec.items():
        print(axes)
        for metric in metrics:
            print("\t", metric)

            key = f"{axes} {metric}"
            print("\t - Concrete:", np.round(concrete_data_agg[key], 3))
            print("\t - Sampled:", np.round(sampled_data_agg[key], 3))
            
def display_data_metrics_comparison_tbl(
    metric_spec,
    concrete_data_agg, sampled_data_agg):
    print("Axis", "&", "Metric", "&",
                  "Genmatch Value", "&",
                  "Maccabee Value", "\\\\ \hline")
    for axes, metrics in metric_spec.items():
        for metric in metrics:
            axis_name = " ".join(axes.split("_")).title()
            key = f"{axes} {metric}"
            print(axis_name, "&", metric, "&",
                  np.round(concrete_data_agg[key], 3), "&",
                  np.round(sampled_data_agg[key], 3), "\\\\ \hline")

def display_aggregated_performance_metrics(metrics):
    keys = sorted(metrics.keys())
    for key in keys:
        print(f"{key}: {np.round(metrics[key], 3)}")

def display_distro(values, title):
    plt.title(title)
    plt.hist(values, density=True)
    plt.show()

def display_raw_performance_metrics(metrics, metric_name, context):
    display_distro(metrics[metric_name],
        f"Distribution of {metric_name} values for {context}")

def _get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def display_sampled_dgps_vs_concrete(
    sampled_run_data, concrete_run_data, metric):

    sampled_runs = sampled_run_data[metric]
    cmap = _get_cmap(len(sampled_runs))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    for i, distro_vals in enumerate(sampled_runs):
        c = cmap(i)
        plt.hist(distro_vals, density=False, alpha=0.25, color=c)
    
    plt.hist(concrete_run_data[metric],
        density=False, color="b", label="Genmatch DGP")
    plt.axvline(x=np.mean(concrete_run_data[metric]), c="b", ymax=1)
    
    plt.xlabel(f"{metric} value")
    plt.ylim((0, 5))
    plt.legend()
    
    plt.subplot(122)
    for i, distro_vals in enumerate(sampled_runs):
        c = cmap(i)
#         plt.errorbar(np.mean(distro_vals), i, fmt="o", xerr=np.std(distro_vals), color="r")
        plt.axvline(x=np.mean(distro_vals), color="r", ymax=0.8)
    plt.axvline(x=np.mean(concrete_run_data[metric]), color="b", label="Genmatch DGP", ymax=0.8)
#     plt.errorbar(np.mean(concrete_run_data[metric]), i//2, fmt="o",
#                  xerr=np.std(concrete_run_data[metric]), color="b", label="Genmatch DGP")
    
    plt.xlabel(f"{metric} value")
    plt.legend()
    
    plt.show()

def plot_dgp_metric_vs_perf(metric_run_level, perf_run_level, color, label):
    mean_metric = np.mean(metric_run_level)
    mean_perf = np.mean(perf_run_level)

    plt.scatter(mean_metric, mean_perf, c=color, label=label)
    interval_data = np.percentile(perf_run_level, [2.5, 97.5]).reshape((2, 1))
    interval_data[0, :] = mean_perf - interval_data[0, :]
    interval_data[1, :] = interval_data[1, :] - mean_perf
    plt.errorbar(mean_metric, mean_perf, interval_data, fmt='none', c=color)
