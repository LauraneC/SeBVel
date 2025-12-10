import numpy as np





# Function to calculate average amplitude per 10x10 degree bin
def calculate_bin_averages(aspect_angles, slope_angles, amplitudes, criterion='mean'):
    bins_aspect = np.arange(0, 370, 10)
    bins_slope = np.arange(0, 90, 10)
    bin_averages = np.zeros((len(bins_slope) - 1, len(bins_aspect) - 1))
    bin_number_of_values = np.zeros((len(bins_slope) - 1, len(bins_aspect) - 1))

    for i in range(len(bins_aspect) - 1):
        for j in range(len(bins_slope) - 1):
            mask = ((aspect_angles >= bins_aspect[i]) & (aspect_angles < bins_aspect[i + 1]) &
                    (slope_angles >= bins_slope[j]) & (slope_angles < bins_slope[j + 1]))
            bin_values = amplitudes[mask].compressed() if np.ma.isMaskedArray(amplitudes) else amplitudes[mask].copy()
            bin_number_of_value = len(bin_values)
            if len(bin_values) > 0:
                if criterion == 'mean':
                    bin_averages[j, i] = np.nanmean(bin_values)
                elif criterion == 'median':
                    bin_averages[j, i] = np.nanmedian(bin_values)
                elif criterion == 'std':
                    bin_averages[j, i] = np.std(bin_values)
                elif criterion == 'MAD':
                    bin_averages[j, i] = stats.median_abs_deviation(bin_values)
                elif criterion == '90':
                    bin_averages[j, i] = np.percentile(bin_values, 90)
                bin_number_of_values[j, i] = bin_number_of_value
            else:
                bin_averages[j, i] = np.nan  # Handle empty bins
                bin_number_of_values[j, i] = np.nan

    return bins_aspect, bins_slope, bin_averages, bin_number_of_values


# Function to create polar plot with bin averages
def create_polar_plot_with_bins(bin_averages, bins_aspect, bins_slope, title, color_map, vmin, vmax, subplot_position,
                                norm='normal', one_plot=False):
    if not one_plot:
        ax = plt.subplot(1, 2, subplot_position, projection='polar')
    else:
        ax = plt.subplot(projection='polar')
    theta, r = np.meshgrid(np.deg2rad(bins_aspect), 80 - bins_slope)

    # Apply logarithmic scale using LogNorm
    if norm == 'log':
        norm = colors.LogNorm(vmin=1, vmax=vmax)
        c = ax.pcolormesh(theta, r, bin_averages, cmap=color_map, norm=norm, shading='auto')
    else:
        c = ax.pcolormesh(theta, r, bin_averages, cmap=color_map, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_title(title, fontsize=16, pad=50)
    plt.colorbar(c, ax=ax, pad=0.1, orientation='vertical').set_label('Seasonal Amplitude (m/y)', fontsize=16)
    r_labels = np.arange(0, 80, 10)  # Adjust step size as needed
    ax.set_yticks(80 - r_labels)  # Position ticks accordingly
    ax.set_yticklabels(r_labels, fontsize=16)  # Label them correctly
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))  # Set the custom angle ticks
    ax.set_xticklabels(['N', 'E', 'S', 'W'], fontsize=16)  # Label them as N, E, S, W

    # Position the y-tick labels more prominently at the top
    for label in ax.get_yticklabels():
        label.set_verticalalignment('bottom')


def plot_hist(data_amplitude, label='EW'):
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4))

    data_list = []
    # Plot histogram with PDF and median for each quadrant
    for quadrant, (low, high) in quadrants.items():
        mask = (aspect_angles >= low) & (aspect_angles < high)
        data_list.append(data_amplitude[mask])

    if len(data_list) > 0:
        north = np.ma.concatenate([data_list[0], data_list[2]])
        # Plot histogram
        bins_number = 500
        ax.hist(north, bins=bins_number, density=True,
                alpha=0.2, color='b', range=(0, vmax), label='North')
        ax.hist(data_list[1], bins=bins_number, density=True, alpha=0.2, color='r',
                range=(0, vmax), label='South')

        ax.set_ylim(0, ymax)

        # Compute statistics for the two groups
        mus, stds = np.mean(data_list[1]), np.std(data_list[1])
        mun, stdn = np.mean(north), np.std(north)

        p68n, p90n = np.percentile(north, 50), np.percentile(north, 90)
        p68s, p90s = np.percentile(data_list[1], 50), np.percentile(data_list[1], 90)

        # Prepare statistics text
        # stats_text = f"South: mean {mus:.1f} m/y ; std {stds:.1f} m/y\n North: mean {mun:.1f} m/y ; std {stdn:.1f} m/y"
        stats_text = f"South: 50% {p68s:.1f} m/y ; 90% {p90s:.1f} m/y\n North: 50% {p68n:.1f} m/y ; 90% {p90n:.1f} m/y"

        # Place the text below the graph using fig.text
        fig.text(0.6, -0.1, stats_text, ha='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xlabel(f"{label} Amplitude [m/y]", fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylabel("Density")

    plt.tight_layout()
    fig.savefig(f"{path_tiff}/Hist_{label}_amplitude_NS.png", dpi=300, bbox_inches='tight')
    plt.show()
    return north, data_list[1]
