import os
from fls.datasets.SamplesDataset import SamplesDataset


class DatasetSet:
    def __init__(self):
        pass

    def get_studiogan_datasets(self, model_names, save_path, dataset_name):
        full_model_names = os.listdir(save_path)

        for model_name in model_names:
            # Retrieves the full name (model dependent in studiogan)

            full_model_name = None
            for name in full_model_names:
                if model_name in name:
                    full_model_name = name
                    break

            if not full_model_name:
                raise Exception(f"Model {model_name} not found in {full_model_names}")

            self.sample_datasets.append(
                SamplesDataset(
                    f"{dataset_name}_{model_name}",
                    path=os.path.join(save_path, full_model_name, "fake"),
                )
            )

    def get_datasets_features(self, datasets, feature_extractor):
        for dataset in datasets:
            feature_extractor.get_features(dataset)
        return

    def create_metric_dicts(self, metrics, feature_extractor):
        # Metric tracking
        metric_vals = {metric.name: {} for metric in metrics}

        for metric in metrics:
            for dataset in self.sample_datasets:
                metric_vals[metric.name][dataset.name] = metric.compute_metric(
                    feature_extractor.get_features(self.train_dataset)[:10000],
                    feature_extractor.get_features(self.train_dataset)[10000:20000],
                    feature_extractor.get_features(self.test_dataset),
                    feature_extractor.get_features(dataset),
                )

        # Create dataset -> {metrics} from metrics->{datasets}
        dataset_vals = {}
        for metric_name, metric_dict in metric_vals.items():
            for dataset_name, metric_val in metric_dict.items():
                dataset_vals[dataset_name] = dataset_vals.get(dataset_name, {})
                dataset_vals[dataset_name][metric_name] = metric_val

        return metric_vals, dataset_vals

    def create_dataset_dict(self):
        self.datasets = [self.train_dataset, self.test_dataset] + self.sample_datasets
        self.datasets = {dataset.name: dataset for dataset in self.datasets}
