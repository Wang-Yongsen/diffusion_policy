import logging, os
import os.path as osp
from tqdm import tqdm
import time
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
from src.poses.utils import load_index_level_in_level2
import torch
from src.utils.bbox_utils import CropResizePad
import pytorch_lightning as pl
from src.dataloader.base_bop import BaseBOP

try:
    from bop_toolkit_lib import dataset_params, inout
except ImportError:
    raise ImportError(
        "Please install bop_toolkit_lib: pip install git+https://github.com/thodan/bop_toolkit.git"
    )

pl.seed_everything(2023)


class BOPTemplate(Dataset):
    def __init__(
        self,
        template_dir,
        obj_ids,
        processing_config,
        level_templates,
        pose_distribution,
        num_imgs_per_obj=50,
        **kwargs,
    ):
        self.template_dir = template_dir
        self.dataset_name = template_dir.split("/")[-2]
        if obj_ids is None:
            obj_ids = [
                int(obj_id[4:10])
                for obj_id in os.listdir(template_dir)
                if osp.isdir(osp.join(template_dir, obj_id))
            ]
            obj_ids = sorted(np.unique(obj_ids).tolist())
            logging.info(f"Found {obj_ids} objects in {self.template_dir}")
        if "onboarding_static" in template_dir or "onboarding_dynamic" in template_dir:
            self.model_free_onboarding = True
        else:
            self.model_free_onboarding = False
        # for HOT3D, we have black objects so we use gray background
        if "hot3d" in template_dir:
            self.use_gray_background = True
            logging.info("Use gray background for HOT3D")
        else:
            self.use_gray_background = False
        self.num_imgs_per_obj = num_imgs_per_obj  # to avoid memory issue
        self.obj_ids = obj_ids
        self.processing_config = processing_config
        self.rgb_transform = T.Compose(
            [
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.proposal_processor = CropResizePad(
            self.processing_config.image_size,
            pad_value=0.5 if self.use_gray_background else 0,
        )
        self.load_template_poses(level_templates, pose_distribution)

    def __len__(self):
        return len(self.obj_ids)

    def load_template_poses(self, level_templates, pose_distribution):
        if pose_distribution == "all":
            self.index_templates = load_index_level_in_level2(level_templates, "all")
        else:
            raise NotImplementedError

    def __getitem__modelbased__(self, idx):
        templates, masks, boxes = [], [], []
        for id_template in self.index_templates:
            image = Image.open(
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}/{id_template:06d}.png"
            )
            boxes.append(image.getbbox())

            mask = image.getchannel("A")
            mask = torch.from_numpy(np.array(mask) / 255).float()
            masks.append(mask.unsqueeze(-1))

            if self.use_gray_background:
                gray_image = Image.new("RGB", image.size, (128, 128, 128))
                gray_image.paste(image, mask=image.getchannel("A"))
                image = gray_image.convert("RGB")
            else:
                image = image.convert("RGB")
            image = torch.from_numpy(np.array(image) / 255).float()
            templates.append(image)

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        templates_croped = self.proposal_processor(images=templates, boxes=boxes)
        masks_cropped = self.proposal_processor(images=masks, boxes=boxes)
        return {
            "templates": self.rgb_transform(templates_croped),
            "template_masks": masks_cropped[:, 0, :, :],
        }

    def __getitem__modelfree__(self, idx):
        templates, masks, boxes = [], [], []
        static_onboarding = True if "onboarding_static" in self.template_dir else False
        if static_onboarding:
            # HOT3D names the two videos with _1 and _2 instead of _up and _down
            if self.dataset_name == "hot3d":
                obj_dirs = [
                    f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_1",
                    f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_2",
                ]
            else:
                obj_dirs = [
                    f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_up",
                    f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_down",
                ]
            num_selected_imgs = self.num_imgs_per_obj // 2  # 100 for 2 videos

            # Objects 34-40 of HANDAL have only one "up" video as these objects are symmetric
            num_video = 0
            for obj_dir in obj_dirs:
                if osp.exists(obj_dir):
                    num_video += 1
            assert (
                num_video > 0
            ), f"No video found for object {self.obj_ids[idx]} in {self.template_dir}"
            if num_video == 1:
                num_selected_imgs = self.num_imgs_per_obj
        else:
            obj_dirs = [
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}",
            ]
            num_selected_imgs = self.num_imgs_per_obj
        for obj_dir in obj_dirs:
            if not osp.exists(obj_dir):
                continue
            obj_dir = Path(obj_dir)
            if self.dataset_name == "hot3d":
                # TODO: currently, only suppport aria and 214-1 stream
                obj_rgbs = sorted(Path(obj_dir).glob("*214-1.[pj][pn][g]"))
                obj_masks = [None for _ in obj_rgbs]
            else:
                # list all rgb
                obj_rgbs = sorted(Path(obj_dir).glob("rgb/*.[pj][pn][g]"))
                # list all masks
                obj_masks = sorted(Path(obj_dir).glob("mask_visib/*.[pj][pn][g]"))
            assert len(obj_rgbs) == len(
                obj_masks
            ), f"rgb and mask mismatch in {obj_dir}"
            
            # If HOT3D + dynamic onboarding, we have the bbox for only the firs timage.
            # therefore, we select the first image only.
            if self.dataset_name == "hot3d" and not static_onboarding:
                selected_idx = [0, 0, 0, 0, 0] # required aggregration top k
            else:
                selected_idx = np.random.choice(
                    len(obj_rgbs), num_selected_imgs, replace=False
                )
            for idx_img in tqdm(selected_idx):
                image = Image.open(obj_rgbs[idx_img])
                if self.dataset_name == "hot3d":
                    json_path = str(obj_rgbs[idx_img]).replace(
                        "image_214-1.jpg", "objects.json"
                    )
                    info = inout.load_json(json_path)
                    obj_id = [k for k in info.keys()][0]
                    bbox = np.int32(info[obj_id][0]["boxes_amodal"]["214-1"])
                    mask = np.ones((image.size[1], image.size[0])) * 255
                else:
                    mask = Image.open(obj_masks[idx_img])
                    image = np.asarray(image) * np.expand_dims(np.asarray(mask) > 0, -1)
                    image = Image.fromarray(image)
                    bbox = mask.getbbox()

                boxes.append(bbox)
                mask = torch.from_numpy(np.array(mask) / 255).float()
                masks.append(mask.unsqueeze(-1))
                image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
                templates.append(image)

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        templates_croped = self.proposal_processor(images=templates, boxes=boxes)
        masks_cropped = self.proposal_processor(images=masks, boxes=boxes)
        return {
            "templates": self.rgb_transform(templates_croped),
            "template_masks": masks_cropped[:, 0, :, :],
        }

    def __getitem__(self, idx):
        if self.model_free_onboarding:
            return self.__getitem__modelfree__(idx)
        else:
            return self.__getitem__modelbased__(idx)


class BaseBOPTest(BaseBOP):
    def __init__(
        self,
        root_dir,
        split,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.split = split
        # dp_split is only required for hot3d dataset.
        self.dataset_name = kwargs.get("dataset_name", None)
        self.dp_split = dataset_params.get_split_params(
            Path(self.root_dir).parent,
            kwargs.get("dataset_name", None),
            split=split,
        )
        # HOT3D test split contains all test images, not only the ones required for evaluation.
        # to speed up the inference, it is faster to only load the images required for evaluation.
        self.load_required_test_images_from_target_file()
        self.load_list_scene(split=split)
        self.load_metaData(reset_metaData=True)
        # shuffle metadata
        self.metaData = self.metaData.sample(frac=1, random_state=2021).reset_index()
        self.rgb_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def load_required_test_images_from_target_file(self) -> None:
        # List all the files in the target directory.
        dataset_dir = Path(self.root_dir)

        # If multiple files are found, use the bop_version to select the correct one.
        target_files = list(dataset_dir.glob("test_targets_bop*.json"))
        if len(target_files) > 1:
            bop_version = "bop19"
            if self.dataset_name in ["hot3d", "hopev2", "handal"]:
                bop_version = "bop24"
            target_files = [f for f in target_files if bop_version in str(f)]
        assert (
            len(target_files) == 1
        ), f"Expected one target file, found {len(target_files)}"
        print(f"Loading target file: {target_files[0]}")
        targets = inout.load_json(str(target_files[0]))
        self.target_images_per_scene = {}
        for item in targets:
            scene_id, im_id = int(item["scene_id"]), int(item["im_id"])
            if scene_id not in self.target_images_per_scene:
                self.target_images_per_scene[scene_id] = []
            self.target_images_per_scene[scene_id].append(im_id)

    def __getitem__(self, idx):
        rgb_path = self.metaData.iloc[idx]["rgb_path"]
        scene_id = self.metaData.iloc[idx]["scene_id"]
        frame_id = self.metaData.iloc[idx]["frame_id"]
        image = Image.open(rgb_path)
        image = self.rgb_transform(image.convert("RGB"))
        return dict(
            image=image,
            scene_id=scene_id,
            frame_id=frame_id,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from omegaconf import DictConfig, OmegaConf
    from torchvision.utils import make_grid, save_image

    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )
    dataset = BOPTemplate(
        template_dir="/home/nguyen/Documents/datasets/gotrack_root_dir/datasets/hot3d/onboarding_static",
        obj_ids=None,
        level_templates=0,
        pose_distribution="all",
        processing_config=processing_config,
    )
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        sample["templates"] = inv_rgb_transform(sample["templates"])
        save_image(sample["templates"], f"./tmp/hot3d_{idx}.png", nrow=7)
