import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging
from huggingface_hub import hf_hub_download

# set level logging
logging.basicConfig(level=logging.INFO)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run_inference",
)
def download(cfg: DictConfig) -> None:
    root_dir = Path(cfg.machine.root_dir) / "datasets"
    os.makedirs(root_dir, exist_ok=True)

    for dataset_name in [
        "handal",
        "hope",
        "hot3d",
    ]:
        # Select the required files based on the dataset name
        if dataset_name in ["hope", "handal"]:
            required_files = ["onboarding_static", "onboarding_dynamic"]
            required_folders = None
        else:
            required_files = None
            required_folders = ["object_ref_aria_dynamic", "object_ref_aria_static"]

        logging.info(f"Downloading {dataset_name}")
        # Download the required files
        if required_files is not None:
            if dataset_name == "hope":
                dataset_dir = root_dir / "hopev2"
            else:
                dataset_dir = root_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            for file_name in required_files:
                file_path = hf_hub_download(
                    repo_id=f"bop-benchmark/{dataset_name}",
                    filename=f"{dataset_name}_{file_name}.zip",
                    repo_type="dataset",
                    local_dir=f"{dataset_dir}",
                )
                logging.info(f"Downloaded to: {file_path}")

                # Unzip the downloaded files
                os.system(
                    f"unzip -o {dataset_dir}/{dataset_name}_{file_name}.zip -d {dataset_dir}"
                )

        # Download the required folders
        if required_folders is not None:
            dataset_dir = root_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            for folder_name in required_folders:
                local_folder_path = f"{root_dir}/{dataset_name}/{folder_name}"
                download_cmd = f"huggingface-cli download bop-benchmark/{dataset_name} --include {folder_name}/*.tar --local-dir {dataset_dir} --repo-type=dataset"
                os.system(download_cmd)
                logging.info(f"Downloaded {folder_name}")

                # Unzip the downloaded files
                files = Path(local_folder_path).glob("*.tar")
                files = sorted(files, key=lambda x: x.name)
                for file in tqdm(files, desc=f"Unzipping {folder_name}"):
                    file_name = file.name.split("-")[-1].split(".")[0]
                    sub_folder = f"{local_folder_path}/{file_name}"
                    os.makedirs(sub_folder, exist_ok=True)

                    unzip_cmd = f"tar -xvf {file} -C {sub_folder}"
                    os.system(unzip_cmd)

        # Rename the folder names for handal and hot3d
        if dataset_name == "handal":
            for onboarding_type in ["static", "dynamic"]:
                os.rename(
                    root_dir / "handal" / onboarding_type,
                    root_dir / "handal" / f"onboarding_{onboarding_type}",
                )
        if dataset_name == "hot3d":
            for onboarding_type in ["static", "dynamic"]:
                os.rename(
                    root_dir / "hot3d" / f"object_ref_aria_{onboarding_type}",
                    root_dir / "hot3d" / f"onboarding_{onboarding_type}",
                )


if __name__ == "__main__":
    download()
