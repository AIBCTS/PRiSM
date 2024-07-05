import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Any
from prism.PRiSM_functions import mlpmask_pred

def compute_dirac_partial_responses(betas, userLambda, x_train, model, device, n_steps):
    x0 = np.zeros((1, x_train.shape[1]))
    if isinstance(model, list):
        y0 = mlpmask_pred(x0, model, device=device)
    elif hasattr(model, 'predict_proba'):
        y0 = model.predict_proba(x0)[:, 1]
    else:
        y0 = model.predict(x0)
    logit_y0 = np.log(y0 / (1 - y0))

    responses = []
    for response in np.where(abs(betas[:, userLambda]) > 0.1)[0]:
        if response < x_train.shape[1]:
            if len(x_train.iloc[:, response].unique()) < n_steps:
                catvals = x_train.iloc[:, response].unique()
                x_in = np.zeros([len(catvals), x_train.shape[1]])
                x_in[:, response] = catvals
            else:
                x_step = np.linspace(min(x_train.iloc[:, response]), max(x_train.iloc[:, response]), n_steps)
                x_in = np.zeros([n_steps, x_train.shape[1]])
                x_in[:, response] = x_step
            if isinstance(model, list):
                pred_y_xi = mlpmask_pred(x_in, model, device=device)
            elif hasattr(model, 'predict_proba'):
                pred_y_xi = model.predict_proba(x_in)[:, 1]
            else:
                pred_y_xi = model.predict(x_in)
            logit_y = np.reshape(np.log(pred_y_xi / (1 - pred_y_xi)) - logit_y0, [len(pred_y_xi), ])
            responses.append((response, logit_y))
    return logit_y0, responses

def compute_lebesgue_partial_responses(betas, userLambda, x_train, model, device, n_steps):
    if isinstance(model, list):
        y0 = mlpmask_pred(x_train, model, device=device)
    elif hasattr(model, 'predict_proba'):
        y0 = model.predict_proba(x_train)[:, 1]
    else:
        y0 = model.predict(x_train)
    logit_y0 = np.mean(np.log(y0 / (1 - y0)))

    responses = []
    for response in np.where(abs(betas[:, userLambda]) > 0.1)[0]:
        if response < x_train.shape[1]:
            if len(x_train.iloc[:, response].unique()) < n_steps:
                catvals = x_train.iloc[:, response].unique()
                logit_y = np.zeros(len(catvals))
                for k in range(len(catvals)):
                    x_in = x_train.copy()
                    x_in.iloc[:, response] = catvals[k]
                    if isinstance(model, list):
                        pred_y_xi = mlpmask_pred(x_in, model, device=device)
                    elif hasattr(model, 'predict_proba'):
                        pred_y_xi = model.predict_proba(x_in)[:, 1]
                    else:
                        pred_y_xi = model.predict(x_in)
                    logit_y[k] = np.mean(np.log(pred_y_xi / (1 - pred_y_xi)) - logit_y0)
            else:
                x_step = np.linspace(min(x_train.iloc[:, response]), max(x_train.iloc[:, response]), n_steps)
                logit_y = np.zeros(n_steps)
                for k in range(n_steps):
                    x_in = x_train.copy()
                    x_in.iloc[:, response] = x_step[k]
                    if isinstance(model, list):
                        pred_y_xi = mlpmask_pred(x_in, model, device=device)
                    elif hasattr(model, 'predict_proba'):
                        pred_y_xi = model.predict_proba(x_in)[:, 1]
                    else:
                        pred_y_xi = model.predict(x_in)
                    logit_y[k] = np.mean(np.log(pred_y_xi / (1 - pred_y_xi)) - logit_y0)
            responses.append((response, logit_y))
    return logit_y0, responses

def compute_dirac_bivariate_responses(betas, userLambda, x_train, x_train0, model, device, bivariate_inputs, n_steps=15):
    # Calculate logit_y0
    x0 = np.zeros((1, x_train.shape[1]))
    if isinstance(model, list):
        y0 = mlpmask_pred(x0, model, device=device)
    elif hasattr(model, 'predict_proba'):
        y0 = model.predict_proba(x0)[:, 1]
    else:
        y0 = model.predict(x0)
    logit_y0 = np.log(y0[0] / (1 - y0[0]))

    bivariate_responses = []

    for pr in np.where(abs(betas[:, userLambda]) > 0.1)[0]:
        if pr >= x_train.shape[1]:
            pr_index = pr - x_train.shape[1]
            pr_i = int(bivariate_inputs[pr_index, 0])
            pr_j = int(bivariate_inputs[pr_index, 1])

            unique_i = x_train.iloc[:, pr_i].nunique()
            unique_j = x_train.iloc[:, pr_j].nunique()

            if unique_i < n_steps:
                x_step0_i = np.sort(x_train0.iloc[:, pr_i].unique())
            else:
                x_step0_i = np.linspace(
                    x_train0.iloc[:, pr_i].min(), x_train0.iloc[:, pr_i].max(), n_steps)

            if unique_j < n_steps:
                x_step0_j = np.sort(x_train0.iloc[:, pr_j].unique())
            else:
                x_step0_j = np.linspace(
                    x_train0.iloc[:, pr_j].min(), x_train0.iloc[:, pr_j].max(), n_steps)

            y_xij = np.zeros((len(x_step0_j), len(x_step0_i)))

            for i, val_i in enumerate(x_step0_i):
                for j, val_j in enumerate(x_step0_j):
                    x_in = np.zeros(x_train.shape[1])
                    x_in[pr_i] = (val_i - x_train0.iloc[:, pr_i].median()
                                  ) / (x_train0.iloc[:, pr_i].std() * 2)
                    x_in[pr_j] = (val_j - x_train0.iloc[:, pr_j].median()
                                  ) / (x_train0.iloc[:, pr_j].std() * 2)

                    if isinstance(model, list):
                        pred_y_xij = mlpmask_pred(
                            x_in.reshape(1, -1), model, device=device)
                    elif hasattr(model, 'predict_proba'):
                        pred_y_xij = model.predict_proba(
                            x_in.reshape(1, -1))[:, 1]
                    else:
                        pred_y_xij = model.predict(x_in.reshape(1, -1))

                    y_xij[j, i] = np.log(
                        pred_y_xij[0] / (1 - pred_y_xij[0])) - logit_y0

            bivariate_responses.append(
                (pr_i, pr_j, x_step0_i, x_step0_j, y_xij))

    return bivariate_responses

def compute_lebesgue_bivariate_responses(betas, userLambda, x_train, x_train0, model, device, bivariate_inputs, n_steps=15):
    # Calculate logit_y0
    if isinstance(model, list):
        y0 = mlpmask_pred(x_train, model, device=device)
    elif hasattr(model, 'predict_proba'):
        y0 = model.predict_proba(x_train)[:, 1]
    else:
        y0 = model.predict(x_train)
    logit_y0 = np.mean(np.log(y0 / (1 - y0)))

    bivariate_responses = []

    for pr in np.where(abs(betas[:, userLambda]) > 0.1)[0]:
        if pr >= x_train.shape[1]:
            pr_index = pr - x_train.shape[1]
            pr_i = int(bivariate_inputs[pr_index, 0])
            pr_j = int(bivariate_inputs[pr_index, 1])

            unique_i = x_train.iloc[:, pr_i].nunique()
            unique_j = x_train.iloc[:, pr_j].nunique()

            if unique_i < n_steps:
                x_step0_i = np.sort(x_train0.iloc[:, pr_i].unique())
            else:
                x_step0_i = np.linspace(
                    x_train0.iloc[:, pr_i].min(), x_train0.iloc[:, pr_i].max(), n_steps)

            if unique_j < n_steps:
                x_step0_j = np.sort(x_train0.iloc[:, pr_j].unique())
            else:
                x_step0_j = np.linspace(
                    x_train0.iloc[:, pr_j].min(), x_train0.iloc[:, pr_j].max(), n_steps)

            y_xij = np.zeros((len(x_step0_j), len(x_step0_i)))

            for i, val_i in enumerate(x_step0_i):
                for j, val_j in enumerate(x_step0_j):
                    x_in = x_train.copy()
                    x_in.iloc[:, pr_i] = (
                        val_i - x_train0.iloc[:, pr_i].median()) / (x_train0.iloc[:, pr_i].std() * 2)
                    x_in.iloc[:, pr_j] = (
                        val_j - x_train0.iloc[:, pr_j].median()) / (x_train0.iloc[:, pr_j].std() * 2)

                    if isinstance(model, list):
                        pred_y_xij = mlpmask_pred(x_in, model, device=device)
                    elif hasattr(model, 'predict_proba'):
                        pred_y_xij = model.predict_proba(x_in)[:, 1]
                    else:
                        pred_y_xij = model.predict(x_in)

                    y_xij[j, i] = np.mean(
                        np.log(pred_y_xij / (1 - pred_y_xij))) - logit_y0

            bivariate_responses.append(
                (pr_i, pr_j, x_step0_i, x_step0_j, y_xij))

    return bivariate_responses

def plot_continuous_response(ax, model, x_train, x_train0, response, logit_y0, logit_y, n_steps):
    x_step0 = np.linspace(min(x_train0.iloc[:, response]), max(
        x_train0.iloc[:, response]), n_steps)
    line, = ax.plot(logit_y, x_step0)
    line_color = line.get_color()

    # Ensure that line_marks0 matches the y-ticks
    line_marks0 = ax.get_yticks()
    line_marks0 = line_marks0[np.where(line_marks0 <= max(x_step0))]
    if min(x_step0) < 0:
        line_marks0 = line_marks0[np.where(line_marks0 >= min(x_step0))]
    else:
        line_marks0 = line_marks0[np.where(line_marks0 >= 0)]
    if len(line_marks0) == 2:
        line_marks0 = np.linspace(line_marks0[0], line_marks0[1], 3)
    ax.set_yticks(line_marks0)

    # Calculate a fixed offsets for annotations
    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0
    x_offset = -0.1
    
    # Adjust the plotting and annotation to match line_marks0
    for mark in line_marks0:
        # Find the corresponding x (logit_y) for each y-tick (mark)
        corresponding_x = np.interp(mark, x_step0, logit_y)
        # Check if corresponding_x is within the range of logit_y and annotate
        if min(x_step0) <= mark <= max(x_step0):
            ax.scatter([corresponding_x], [mark], marker="o", color=line_color)
            if mark.is_integer():
                annotation = f"{int(mark)}"
            else:
                annotation = f"{mark:.1f}".rstrip('0').rstrip('.')
            ax.annotate(annotation, (corresponding_x, mark), xytext=(
                corresponding_x + x_offset, mark + y_offset), ha='center', va='center', color='black')

    ax.axvline(0, color="black", alpha=0.5)
    ax.set_ylabel(x_train.columns[response][:16],
                  rotation=90, loc="center", labelpad=5)
    ax.yaxis.tick_right()

def plot_categorical_response(ax, model, x_train, data, response, logit_y0, logit_y):
    ax.axvline(0, color="black", alpha=0.5)
    ylabel_text = x_train.columns[response][:16]
    ax.set_ylabel(ylabel_text, rotation=90, loc="center", labelpad=5)
    ax.set_yticks([])
    y_value = 0.5

    # plot series
    line, = ax.plot(logit_y, np.full(len(logit_y), y_value))
    line_color = line.get_color()

    # series annotations at each non-zero value
    for i, value in enumerate(logit_y):
        if value != 0:
            ax.scatter(value, y_value, marker="|", color=line_color)
            ax.annotate(data[x_train.columns[response]].unique()[i], (value, y_value), xytext=(value, y_value + 0.003),
                        ha='center', va='bottom')

def plot_categorical_categorical(ax, x_train, pr_i, pr_j, x_step0_i, x_step0_j, y_xij):
    im = ax.imshow(y_xij, cmap='viridis', aspect='auto', origin='lower')
    ax.set_xticks(range(len(x_step0_i)))
    ax.set_yticks(range(len(x_step0_j)))

    # Format x-axis tick labels
    x_labels = [f"{val:g}" if isinstance(
        val, (int, float)) else str(val) for val in x_step0_i]
    ax.set_xticklabels(x_labels)

    # Format y-axis tick labels
    y_labels = [f"{val:g}" if isinstance(
        val, (int, float)) else str(val) for val in x_step0_j]
    ax.set_yticklabels(y_labels)

    for i in range(len(x_step0_j)):
        for j in range(len(x_step0_i)):
            text = ax.text(j, i, f"{y_xij[i, j]:.2f}",
                           ha="center", va="center", color="black",
                           bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.5))

    ax.set_xlabel(x_train.columns[pr_i])
    ax.set_ylabel(x_train.columns[pr_j])
    plt.colorbar(im, ax=ax, label='Log Odds Ratio')

def plot_continuous_continuous(ax, x_train, pr_i, pr_j, x_step0_i, x_step0_j, y_xij):
    X, Y = np.meshgrid(x_step0_i, x_step0_j)
    contour_heatmap = ax.contourf(X, Y, y_xij, cmap='viridis', levels=20)
    contour_lines = ax.contour(X, Y, y_xij, colors='white', alpha=0.5, levels=10)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    ax.set_xlabel(x_train.columns[pr_i])
    ax.set_ylabel(x_train.columns[pr_j])
    plt.colorbar(contour_heatmap, ax=ax, label='Log Odds Ratio')

def plot_mixed_response(ax, x_train, x_train0, pr_i, pr_j, x_step0_i, x_step0_j, y_xij, categorical_threshold):

    def round_to_n_significant(x, n=2):
        if x == 0:
            return 0
        return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

    is_categorical_i = len(x_step0_i) < categorical_threshold

    if is_categorical_i:
        categorical_var = pr_i
        continuous_var = pr_j
        x_categorical = x_step0_i
        x_continuous = x_step0_j
    else:
        categorical_var = pr_j
        continuous_var = pr_i
        x_categorical = x_step0_j
        x_continuous = x_step0_i

    # Plot lines for each categorical value
    for i, val in enumerate(x_categorical):
        if is_categorical_i:
            y_values = y_xij[:, i]
        else:
            y_values = y_xij[i, :]

        label = f"{val:g}" if isinstance(val, (int, float)) else str(val)
        line, = ax.plot(y_values, x_continuous, label=label)
        line_color = line.get_color()

    # Set y-axis ticks, rounding to 2 significant digits
    y_ticks = ax.get_yticks()
    y_ticks = y_ticks[(y_ticks >= min(x_continuous)) &
                      (y_ticks <= max(x_continuous))]
    min_y_ticks = 3
    if len(y_ticks) < min_y_ticks:
        y_ticks = np.linspace(min(x_continuous), max(
            x_continuous), 2*min_y_ticks+1)
        y_ticks = [round_to_n_significant(y) for y in y_ticks[1:-1:2]]

    ax.set_yticks(y_ticks)

    # Format y-axis labels
    y_labels = [f"{val:g}" if isinstance(
        val, (int, float)) else str(val) for val in y_ticks]
    ax.set_yticklabels(y_labels)

    # Add scatter marks at y-tick intersections for each line
    for i, val in enumerate(x_categorical):
        if is_categorical_i:
            y_values = y_xij[:, i]
        else:
            y_values = y_xij[i, :]

        line_color = ax.get_lines()[i].get_color()

        for y_tick in y_ticks:
            if min(x_continuous) <= y_tick <= max(x_continuous):
                x_value = np.interp(y_tick, x_continuous, y_values)
                ax.scatter(x_value, y_tick, marker="o", color=line_color)

    ax.axvline(0, color="black", alpha=0.5)
    ax.set_xlabel("Log Odds Ratio")
    ax.set_ylabel(x_train.columns[continuous_var][:16])

    # Add legend
    ax.legend(title=x_train.columns[categorical_var], fontsize=8, title_fontsize=10,
              loc='lower left', bbox_to_anchor=(0.05, 0.05), borderaxespad=0.)

    # Move y-axis to the right
    ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")

def plot_bivariate_response(ax, x_train, pr_i, pr_j, x_step0_i, x_step0_j, y_xij, categorical_threshold):
    is_categorical_i = len(x_step0_i) < categorical_threshold
    is_categorical_j = len(x_step0_j) < categorical_threshold

    if is_categorical_i and is_categorical_j:
        plot_categorical_categorical(ax, x_train, pr_i, pr_j, x_step0_i, x_step0_j, y_xij)
    elif is_categorical_i or is_categorical_j:
        plot_mixed_response(ax, x_train, pr_i, pr_j, x_step0_i, x_step0_j, y_xij, categorical_threshold)
    else:
        plot_continuous_continuous(ax, x_train, pr_i, pr_j, x_step0_i, x_step0_j, y_xij)

    plt.tight_layout()

def prnomogram_refactor(betas: List[float], userLambda: float, x_train0: pd.DataFrame, x_train: pd.DataFrame, data: pd.DataFrame, model: Any, bivariate_inputs: List[int], n_steps: int = 15, sd_scale: int = 2, method: str = "dirac", device: str = "cpu", categorical_threshold: int = 15) -> None:
    # Compute partial responses
    if method.lower() == "dirac":
        logit_y0, responses = compute_dirac_partial_responses(
            betas, userLambda, x_train, model, device, n_steps)
        bivariate_responses = compute_dirac_bivariate_responses(
            betas, userLambda, x_train, x_train0, model, device, bivariate_inputs, n_steps)
    elif method.lower() == "lebesgue":
        logit_y0, responses = compute_lebesgue_partial_responses(
            betas, userLambda, x_train, model, device, n_steps)
        bivariate_responses = compute_lebesgue_bivariate_responses(
            betas, userLambda, x_train, x_train0, model, device, bivariate_inputs, n_steps)
    else:
        raise ValueError("Invalid method: choose 'dirac' or 'lebesgue'")

    # Determine the number of subplots needed
    num_univariate = len(responses)
    num_mixed = sum(1 for _, _, x_i, x_j, _ in bivariate_responses
                    if (len(x_i) < categorical_threshold) != (len(x_j) < categorical_threshold))
    num_subplots = num_univariate + num_mixed

    # Create the main figure
    subfig_height = 1.8
    fig_height = subfig_height * num_subplots
    nomo = plt.figure(figsize=(8, fig_height))
    gs = gridspec.GridSpec(num_subplots, 1, height_ratios=[
                           subfig_height] * num_subplots)

    # Plot univariate responses
    subplot_index = 0
    all_x_values = []

    for i, (response, logit_y) in enumerate(responses):
        ax = nomo.add_subplot(gs[subplot_index])
        all_x_values.extend(logit_y)
        if i == 0:
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.spines['top'].set_position(('outward', 10))
            ax.set_xlabel("Log Odds Ratio")
        else:
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(labelbottom=False)

        if len(x_train.iloc[:, response].unique()) < categorical_threshold:
            plot_categorical_response(
                ax, model, x_train, data, response, logit_y0, logit_y)
        else:
            plot_continuous_response(
                ax, model, x_train, x_train0, response, logit_y0, logit_y, n_steps)

        subplot_index += 1

    # Plot mixed bivariate responses
    for pr_i, pr_j, x_step0_i, x_step0_j, y_xij in bivariate_responses:
        is_categorical_i = len(x_step0_i) < categorical_threshold
        is_categorical_j = len(x_step0_j) < categorical_threshold

        if is_categorical_i != is_categorical_j:  # Mixed case
            ax = nomo.add_subplot(gs[subplot_index])
            plot_mixed_response(ax, x_train, x_train0, pr_i, pr_j,
                                x_step0_i, x_step0_j, y_xij, categorical_threshold)
            all_x_values.extend(y_xij.flatten())
            subplot_index += 1

    # Set common x-axis limits
    if all_x_values:
        min_x = min(all_x_values)
        max_x = max(all_x_values)
        x_padding_ratio = 0.05
        for ax in nomo.axes:
            ax.set_xlim(min_x-(x_padding_ratio*(max_x-min_x)),
                        max_x+(x_padding_ratio*(max_x-min_x)))

    # Adjust the title positioning
    title_height = 1  # Height reserved for title in inches
    fig_top = 1 - (title_height / nomo.get_figheight())

    nomo.suptitle(
        "Nomogram of univariate and mixed bivariate partial responses", y=1)

    # Adjust layout
    nomo.tight_layout()
    nomo.subplots_adjust(top=fig_top, hspace=0)
    nomo.show()

    # Plot categorical-categorical and continuous-continuous bivariate responses
    non_mixed_responses = [(pr_i, pr_j, x_step0_i, x_step0_j, y_xij)
                           for pr_i, pr_j, x_step0_i, x_step0_j, y_xij in bivariate_responses
                           if (len(x_step0_i) < categorical_threshold) == (len(x_step0_j) < categorical_threshold)]

    if non_mixed_responses:
        num_bivariate = len(non_mixed_responses)
        binomo = plt.figure(figsize=(8, 5 * num_bivariate))

        for i, (pr_i, pr_j, x_step0_i, x_step0_j, y_xij) in enumerate(non_mixed_responses):
            ax = binomo.add_subplot(num_bivariate, 1, i+1)
            plot_bivariate_response(
                ax, x_train, pr_i, pr_j, x_step0_i, x_step0_j, y_xij, categorical_threshold)

        # Adjust the title positioning for non-mixed plot
        title_height = 0.5  # Height reserved for title in inches
        fig_top = 1 - (title_height / binomo.get_figheight())

        binomo.suptitle(
            "Nomogram of non-mixed bivariate partial responses", y=1)

        # Adjust layout
        binomo.tight_layout()
        binomo.subplots_adjust(top=fig_top, hspace=0.2)
        binomo.show()
