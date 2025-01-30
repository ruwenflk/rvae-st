import sys, os
import numpy as np
import pca_tsne

module_path = os.path.abspath(os.path.join(".."))
sys.path.append(module_path + "/src")

from config.Config import Config


if __name__ == "__main__":
    # gpu_selection.select_gpu_with_lowest_memory()
    config = Config()
    # models = ["ti-rvae", "wavegan", "timegan", "timevae"]
    models = ["wavegan", "wavegan2", "ti-rvae", "timegan", "rcgan", "timevae2"]
    working_space = "/workspace/git/sydapro_paper/paper_stuff"
    datasets = ["ett", "ecg", "lenze"]
    # datasets = ["ecg"]
    seq_lengths = [100, 300, 500, 1000]
    # ../output/experiments/ti-rvae/ecg/config.conf

    for model_name in models:
        for dataset in datasets:
            for seq in seq_lengths:
                print(dataset, seq)
                output_path = f"../output/experiments/{model_name}/{dataset}"
                config.select_output_config_by_args(
                    output_path, "config.conf", str(seq)
                )

                n_samples = 250
                if dataset == "ecg":
                    n_samples = 1000
                subset = "train"

                data, model = pca_tsne.get_data_and_model(config)

                chunks_original, chunks_synthetic = (
                    pca_tsne.get_prepared_reduced_chunks(
                        model, data, config, n_samples, subset
                    )
                )

                pca_real, pca_synth, tsne_results = pca_tsne.get_pca_and_tsne_results(
                    chunks_original, chunks_synthetic
                )

                # plot_pca_and_tsne(pca_real, pca_synth, tsne_results)
                fig_pca = pca_tsne.plot_pca(config, pca_real, pca_synth)
                fig_tsne = pca_tsne.plot_tsne(config, tsne_results)

                fig_pca.savefig(
                    f"../output/experiments/pca_tsne/pca_{model_name}_{dataset}_{seq}.png"
                )

                fig_tsne.savefig(
                    f"../output/experiments/pca_tsne/tsne_{model_name}_{dataset}_{seq}.png"
                )
