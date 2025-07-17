import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch

from recbole.data import KnowledgeBasedDataLoader
from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer, set_color
from recbole.utils.case_study import full_sort_topk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Case study example")
    parser.add_argument(
        "model_files",
        type=str,
        nargs="+",
        help="The path of the model files pre-trained by recbole",
    )
    parser.add_argument(
        "--recipes_with_cf_wf",
        type=str,
        default="recipes_with_cf_wf.csv",
        help="The path to the recipes with CF and WF",
    )
    parser.add_argument(
        "--plots_path", type=str, default="plots", help="The path to save the plots"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="The id of gpu")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=50_000,
        help="The batch size of evaluation",
    )
    parser.add_argument("--skip_eval", action="store_true", help="Skip the evaluation")
    parser.add_argument(
        "--CF_WF_per_serving_size",
        action="store_true",
        help="Calculate CF and WF per serving size",
    )
    args, _ = parser.parse_known_args()

    os.makedirs(args.plots_path, exist_ok=True)

    if args.CF_WF_per_serving_size:
        sustainability_df_cf_column = "CF_recipe"
        sustainability_df_wf_column = "WF_recipe"
    else:
        sustainability_df_cf_column = "recipe_CF_kg"
        sustainability_df_wf_column = "recipe_WF_kg"

    sustainability_df = pl.read_csv(args.recipes_with_cf_wf, separator="\t")

    topk_cf_wf_dict_filename = "topk_cf_wf_per_model.pkl"
    if not os.path.exists(topk_cf_wf_dict_filename):
        topk_cf_wf_per_model = {}
        for model_file in args.model_files:
            print(f"Loading model from {model_file}")
            config, model, dataset, train_data, valid_data, test_data = (
                load_data_and_model(model_file=model_file)
            )
            config.compatibility_settings()
            config["eval_batch_size"] = args.eval_batch_size
            config["gpu_id"] = args.gpu_id
            config["metrics"] += [
                "giniindex",
                "shannonentropy",
                "itemcoverage",
                "tailpercentage",
                "novelty",
            ]  # , "serendipity"]

            if not args.skip_eval:
                trainer = get_trainer(config["MODEL_TYPE"], config["model"])(
                    config, model
                )
                trainer.resume_checkpoint(model_file)
                if isinstance(train_data, KnowledgeBasedDataLoader):
                    train_data.dataset = train_data._dataset

                trainer.eval_collector.data_collect(train_data)

                best_valid_score, best_valid_result = trainer._valid_epoch(
                    valid_data, show_progress=True
                )
                test_result = trainer.evaluate(
                    test_data, load_best_model=False, show_progress=True
                )

                from logging import getLogger

                logger = getLogger()
                logger.info(
                    set_color("best valid ", "yellow") + f": {best_valid_result}"
                )
                logger.info(set_color("test result", "yellow") + f": {test_result}")

                result = {
                    "best_valid_score": best_valid_score,
                    "valid_score_bigger": config["valid_metric_bigger"],
                    "best_valid_result": best_valid_result,
                    "test_result": test_result,
                }

                print(result)

                result_filepath = os.path.join(args.plots_path, "topk_result_per_model.pkl")
                if os.path.exists(result_filepath):
                    with open(result_filepath, "rb") as f:
                        existing_result = pickle.load(f)
                    existing_result[model.__class__.__name__] = result
                    result = existing_result
                else:
                    result = {model.__class__.__name__: result}

                with open(result_filepath, "wb") as f:
                    pickle.dump(result, f)

            topk_lists = []
            users = test_data.user_df[test_data.uid_field]
            for user_batch in torch.split(users, args.eval_batch_size):
                _, batch_topk_lists = full_sort_topk(
                    user_batch, model, test_data, 10, device=model.device
                )
                topk_lists.append(batch_topk_lists.cpu())

            topk_lists = torch.cat(topk_lists, dim=0)

            cf_map = torch.full((dataset.item_num,), fill_value=torch.nan, dtype=float)
            for recipe_id, cf in zip(
                sustainability_df["recipe_id"].cast(pl.String),
                sustainability_df[sustainability_df_cf_column].to_torch(),
            ):
                if recipe_id in dataset.field2token_id[dataset.iid_field]:
                    cf_map[dataset.field2token_id[dataset.iid_field][recipe_id]] = cf

            wf_map = torch.full((dataset.item_num,), fill_value=torch.nan, dtype=float)
            for recipe_id, wf in zip(
                sustainability_df["recipe_id"].cast(pl.String),
                sustainability_df[sustainability_df_wf_column].to_torch(),
            ):
                if recipe_id in dataset.field2token_id[dataset.iid_field]:
                    wf_map[dataset.field2token_id[dataset.iid_field][recipe_id]] = wf

            model_name = model.__class__.__name__

            topk_cf = cf_map[topk_lists]
            topk_wf = wf_map[topk_lists]

            topk_cf_wf_per_model[model_name] = (topk_cf.numpy(), topk_wf.numpy())

        with open(topk_cf_wf_dict_filename, "wb") as f:
            pickle.dump(topk_cf_wf_per_model, f)
    else:
        with open(topk_cf_wf_dict_filename, "rb") as f:
            topk_cf_wf_per_model = pickle.load(f)

    models_list = ["KGAT", "LightGCN", "Pop"]  # list(topk_cf_wf_per_model.keys())
    cf_threshold = 10
    wf_threshold = 35000
    cf_label = "CFP (Kgs CO$_{2}$eq/kg)"
    wf_label = "WFP (Liters water/kg)"

    samples, hue = [], []
    heatmap_cf_list, heatmap_wf_list = [], []
    for model_name in models_list:
        topk_cf, topk_wf = topk_cf_wf_per_model[model_name]
        # Compute column‐wise average ignoring nan values (each column = position in recommendation list)
        cf_avg = np.nanmean(topk_cf, axis=0)
        wf_avg = np.nanmean(topk_wf, axis=0)
        heatmap_cf_list.append(cf_avg)
        heatmap_wf_list.append(wf_avg)

        model_samples = np.vstack([topk_cf.ravel(), topk_wf.ravel()]).T
        model_samples = model_samples[np.logical_not(np.isnan(model_samples).any(1))]
        hue.extend([model_name] * model_samples.shape[0])
        samples.append(model_samples)

    # Generate heatmaps for CF and WF with shared y-axis in the same figure
    heatmap_cf = np.vstack(heatmap_cf_list)
    heatmap_wf = np.vstack(heatmap_wf_list)
    annot_size = 10
    title_size = 13
    cbar_size = 10
    tick_fontsize = 10
    xlabel_size = 10

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 3))
    sns.heatmap(
        heatmap_cf,
        ax=ax1,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_size},
        yticklabels=models_list,
        xticklabels=range(1, topk_cf.shape[1] + 1),
        linewidth=0.5,
    )
    # ax1.set_title('CF (Kgs CO$_{2}$eq/g) Avg. Across Top-K Positions', fontsize=title_size)
    # ax1.set_xlabel('Position', fontsize=xlabel_size)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    # For the WF heatmap, use scientific notation formatting for cell annotations
    sns.heatmap(
        heatmap_wf,
        ax=ax2,
        cmap="magma",
        annot=True,
        fmt=".2f",
        linewidth=0.5,
        annot_kws={"size": annot_size},
        yticklabels=models_list,
        xticklabels=range(1, topk_cf.shape[1] + 1),
    )
    # ax2.set_title('WF (Liters water/g) Avg. Across Top-K Positions', fontsize=title_size)
    ax2.set_xlabel("Position", fontsize=xlabel_size)
    # Configure the WF colorbar to display values in exponential notation
    cb = ax2.collections[0].colorbar
    cb.formatter.set_scientific(True)
    exponent_limit = np.floor(np.log10(cb.vmax)) + 1
    cb.formatter.set_powerlimits((exponent_limit, exponent_limit))
    cb.update_ticks()

    ax1.collections[0].colorbar.ax.tick_params(labelsize=cbar_size)
    cb.ax.tick_params(labelsize=cbar_size)

    for ax in (ax1, ax2):
        plt.setp(ax.get_xticklabels(), fontsize=tick_fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)

    for t in ax2.texts:
        t.set_text(f"{float(t.get_text()) / (10**exponent_limit):.1f}".split("e")[0])

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.plots_path, "cf_wf_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    samples = np.vstack(samples)
    hue = np.array(hue)

    # Remove outliers
    outliers_mask = np.logical_and(
        samples[:, 0] < cf_threshold, samples[:, 1] < wf_threshold
    )
    samples = samples[outliers_mask]
    hue = hue[outliers_mask]

    grid = sns.jointplot(x=samples[:, 0], y=samples[:, 1], hue=hue)
    grid.ax_joint.set_xlabel(cf_label)
    grid.ax_joint.set_ylabel(wf_label)
    plt.tight_layout()
    grid.ax_joint.legend(title="Model", fontsize=10)
    plt.savefig(
        os.path.join(args.plots_path, "cf_wf_jointplot_hummus.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.model_files[0]
    )

    cf_map = torch.full((dataset.item_num,), fill_value=torch.nan, dtype=float)
    for recipe_id, cf in zip(
        sustainability_df["recipe_id"].cast(pl.String),
        sustainability_df[sustainability_df_cf_column].to_torch(),
    ):
        if recipe_id in dataset.field2token_id[dataset.iid_field]:
            cf_map[dataset.field2token_id[dataset.iid_field][recipe_id]] = cf

    wf_map = torch.full((dataset.item_num,), fill_value=torch.nan, dtype=float)
    for recipe_id, wf in zip(
        sustainability_df["recipe_id"].cast(pl.String),
        sustainability_df[sustainability_df_wf_column].to_torch(),
    ):
        if recipe_id in dataset.field2token_id[dataset.iid_field]:
            wf_map[dataset.field2token_id[dataset.iid_field][recipe_id]] = wf

    users_test_pos_iids = test_data.dataset.history_item_matrix()[0]
    test_pos_iids_cf = cf_map[users_test_pos_iids]
    test_pos_iids_wf = wf_map[users_test_pos_iids]

    test_pos_iids_samples = torch.vstack(
        [test_pos_iids_cf.ravel(), test_pos_iids_wf.ravel()]
    ).T
    test_pos_iids_samples = test_pos_iids_samples[
        torch.logical_not(torch.isnan(test_pos_iids_samples).any(1))
    ]
    test_pos_iids_samples = test_pos_iids_samples.numpy()

    outliers_mask = np.logical_and(
        test_pos_iids_samples[:, 0] < cf_threshold,
        test_pos_iids_samples[:, 1] < wf_threshold,
    )
    test_pos_iids_samples = test_pos_iids_samples[outliers_mask]
    test_pos_iids_hue = ["Test Interactions"] * test_pos_iids_samples.shape[0]

    palette = dict(
        zip(
            models_list + ["Test Interactions"],
            sns.color_palette("tab10", n_colors=len(models_list) + 1),
        )
    )

    scatter_grid = sns.jointplot(
        x=test_pos_iids_samples[:, 0],
        space=0.1,
        y=test_pos_iids_samples[:, 1],
        hue=test_pos_iids_hue,
        palette=palette,
    )
    sns.scatterplot(
        x=samples[:, 0],
        y=samples[:, 1],
        hue=hue,
        alpha=0.9,
        ax=scatter_grid.ax_joint,
        palette=palette,
    )
    sns.kdeplot(
        x=samples[:, 0],
        hue=hue,
        fill=True,
        ax=scatter_grid.ax_marg_x,
        legend=False,
        palette=palette,
    )
    sns.kdeplot(
        y=samples[:, 1],
        hue=hue,
        fill=True,
        ax=scatter_grid.ax_marg_y,
        legend=False,
        palette=palette,
    )
    scatter_grid.ax_joint.set_xlabel(cf_label)
    scatter_grid.ax_joint.set_ylabel(wf_label)
    scatter_legend = scatter_grid.ax_joint.legend(
        title="Test Interactions\nand\nModel Recommendations", fontsize=10
    )
    plt.setp(scatter_legend.get_title(), multialignment="center")
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.plots_path, "cf_wf_jointplot_pos_iid_hummus.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
