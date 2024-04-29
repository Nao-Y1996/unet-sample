import os
import datetime
import json
import numpy as np
from PIL import ImageColor
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


def extract_values_by_key(origins: list[dict], key: str, value_list: list[str]):
    """"
    Parameters
    ----------
    origins: list[dict]
        list of dictionaries
    key: str
        key of value to extract
    value_list: list[str]
        list of values to extract
    """
    new_list = []
    for origin in origins:
        if origin[key] in value_list:
            new_list.append(origin)
    return new_list


def exclude_values_by_key(origins: list[dict], key: str, value_list: list[str]):
    """"
    Parameters
    ----------
    origins: list[dict]
        list of dictionaries
    key: str
        key of value to exclude
    value_list: list[str]
        list of values to exclude
    """
    new_list = []
    for origin in origins:
        if origin[key] not in value_list:
            new_list.append(origin)
    return new_list


class CustomCocoDataset(Dataset):
    def __init__(self, image_dir: str,
                 annotation_file: str,
                 data_transforms=None,
                 label_transforms=None,
                 include_files: list[str] = None,
                 exclude_files: list[str] = None,
                 include_categories: list[str] = None,
                 exclude_categories: list[str] = None, ):
        """
        constructor of CustomCocoDataset.
        When include_files is given, only the images with the file names in the list are used.
        When exclude_files is given, the images with the file names in the list are excluded.
        When include_categories is given, only the images with the category names in the list are used.
        If include_files and exclude_files both contain the same file name, that file will be excluded from the dataset.

        Attributes
        ----------
        image_dir: str
            path to the image directory
        annotation_file: str
            path to the annotation json file
        data_transforms:
            transform to be applied to the image
        label_transforms:
            transform to be applied to the mask
        include_files: list[str]
            list of file names to include
        exclude_files: list[str]
            list of file names to exclude
        include_categories: list[str]
            list of category names to include
        exclude_categories: list[str]
            list of category names to exclude

        """

        if ((include_files is not None) or
                (exclude_files is not None) or
                (include_categories is not None) or
                (exclude_categories is not None)):
            with open(annotation_file) as f:
                annotation_dict = json.load(f)

            # filter by file names
            if include_files is not None or exclude_files is not None:
                # 画像一覧の修正
                new_images = annotation_dict["images"]
                if include_files is not None:
                    new_images = extract_values_by_key(new_images, key="file_name", value_list=include_files)
                if exclude_files is not None:
                    new_images = exclude_values_by_key(new_images, key="file_name", value_list=exclude_files)
                # アノテーション一覧の修正
                new_image_ids = [image["id"] for image in new_images]
                new_annotations = extract_values_by_key(annotation_dict["annotations"], key="image_id",
                                                        value_list=new_image_ids)

                # アノテーション情報を更新
                annotation_dict = {"images": new_images, "annotations": new_annotations,
                                   "categories": annotation_dict["categories"]}

            # filter by category names
            if include_categories is not None or exclude_categories is not None:

                # カテゴリ一覧の修正
                new_categories = annotation_dict["categories"]
                if include_categories is not None:
                    new_categories = extract_values_by_key(new_categories, key="name", value_list=include_categories)
                if exclude_categories is not None:
                    new_categories = exclude_values_by_key(new_categories, key="name", value_list=exclude_categories)

                # アノテーション一覧の修正
                new_category_ids = [category["id"] for category in new_categories]
                new_annotations = extract_values_by_key(annotation_dict["annotations"], key="category_id",
                                                        value_list=new_category_ids)

                # 画像一覧の修正
                new_image_ids = [annotation["image_id"] for annotation in new_annotations]
                new_images = extract_values_by_key(annotation_dict["images"], key="id", value_list=new_image_ids)

                # 新しいアノテーションファイルを作成
                # category_names = [category["name"] for category in new_categories]
                annotation_dict = {"images": new_images, "annotations": new_annotations, "categories": new_categories}

            self.__colors = None
            if "color" in annotation_dict["categories"][0].keys():
                self.__colors = [ImageColor.getrgb(category['color']) for category in annotation_dict["categories"]]
            now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            new_ann_file = annotation_file.replace(".json", f"new_{now}.json")
            with open(new_ann_file, "w") as f:
                json.dump(annotation_dict, f)
            self.coco = COCO(str(new_ann_file))
            os.remove(new_ann_file)
        else:
            self.coco = COCO(str(annotation_file))

        self.image_dir = image_dir
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        index: int
            index of the image

        Returns
        -------
        img: torch.Tensor
            image tensor
        target: dict
            dictionary containing masks, origin_height, origin_width
            mask is a tensor of shape (n_batches, n_classes, height, width)
        """
        # 画像IDを取得
        img_id = self.ids[index]
        height = self.coco.loadImgs(img_id)[0]["height"]
        width = self.coco.loadImgs(img_id)[0]["width"]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        # 画像IDに対応するアノテーションID一覧を取得（1枚の画像に複数のアノテーションがある）
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # アノテーションID一覧から画像に紐ずくアノテーション一覧を取得
        coco_anns = self.coco.loadAnns(ann_ids)

        category_ids = self.coco.getCatIds()

        self.cat_id2idx = {}
        self.cat_name2idx = {}
        for idx, category in enumerate(self.coco.loadCats(category_ids)):
            self.cat_id2idx[category["id"]] = idx
            self.cat_name2idx[category["name"]] = idx

        # masks: (num_of_category_mask, height, width)
        masks = np.zeros(shape=(len(category_ids), height, width), dtype=bool)
        # アノテーションをカテゴリごとに結合してカテゴリマスクを作成
        for ann in coco_anns:
            category_id: int = int(ann['category_id'])
            index_of_cat = self.cat_id2idx[category_id]
            category_mask = masks[index_of_cat]  # get category mask
            category_mask = np.logical_or(category_mask, self.coco.annToMask(ann))  # update category mask
            masks[index_of_cat] = category_mask  # put category mask back
        masks = torch.Tensor(masks.astype(np.float32))

        img: torch.Tensor = read_image(os.path.join(self.image_dir, file_name))

        if self.data_transforms is not None:
            img = self.data_transforms(img)
        if self.label_transforms is not None:
            masks = self.label_transforms(masks)

        target = {
            "masks": masks,
            "file_name": file_name,
            "origin_height": height,
            "origin_width": width
        }

        return img, target

    def __len__(self) -> int:
        return len(self.ids)

    def get_colors(self, color_type: str):
        """
        get color of each category.

        Parameters
        ----------
        color_type: str
            'rgb' or 'bgr'

        Returns
        -------
        colors: list[tuple]
            list of rgb color. ex. [(255, 0, 0), (0, 255, 0)]
        """
        if self.__colors is None:
            raise ValueError("annotasion file not contains color information")
        if color_type == "rgb":
            return self.__colors  # (r, g, b)
        elif color_type == "bgr":
            return [(color[2], color[1], color[0]) for color in self.__colors]  # (b, g, r)
        else:
            raise ValueError("color_type should be 'rgb' or 'bgr'")

    def get_category_name(self, category_id: int):
        categories = self.coco.loadCats(category_id)
        return categories[0]['name']

    def get_categories(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        return [category['name'] for category in categories]


if __name__ == '__main__':

    import cv2
    from torchvision import transforms
    from torch.utils.data import DataLoader

    from visualize.mask_utils import draw_mask_contours
    from visualize.color import BGR_COLORS

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((768, 768)),
        transforms.ToTensor(),
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize((768, 768)),
    ])

    image_dir = "/Users/nao.yamada/personal/unet-sample/datasets/coco-2017/validation/data"
    annotation_file = "/Users/nao.yamada/personal/unet-sample/datasets/coco-2017/validation/labels.json"
    dataset = CustomCocoDataset(image_dir=image_dir,
                                 annotation_file=annotation_file,
                                 data_transforms=img_transforms,
                                 label_transforms=mask_transforms,
                                 include_files=["000000001675.jpg", "000000004795.jpg", "000000007386.jpg"],
                                 exclude_files=["000000004795.jpg"],
                                 include_categories=["person", "cat", "dog"],
                                 exclude_categories=["person"]
                                 )

    print(f"dataset length = {len(dataset)}")

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for img, targets in data_loader:
        img = (img[0] * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        masks = targets["masks"][0]
        masks = masks.to('cpu').detach().numpy().copy()
        origin_height = targets["origin_height"][0]
        origin_width = targets["origin_width"][0]
        colors = list(BGR_COLORS.values())  # dataset.get_colors(color_type="bgr")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask0 = masks[0]
        mask1 = masks[1]
        image = draw_mask_contours(masks=masks, origin_img=img, colors=colors)

        print(f"image shape = {img.shape}")
        print(f"mask shape = {masks.shape}")
        print(f"colors = {colors}")
