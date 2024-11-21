import os
from glob import glob
import pandas as pd
import json
from load_parquet_files import get_df_from_parquet_files


def create_test_file_from_vqa_rad_json(gt_file="/local2/shared_data/VQA-RAD/test.json",
                                       base_image_dir='/local2/shared_data/VQA-RAD/images',
                                       image_field="image",
                                       test_file_name="VQA_RAD_v1",
                                       prompt_type="q1"):
    with open(gt_file, "r") as f:
        gt_anns = json.load(f)
    for ann in gt_anns[:]:
        ann["prompt"] = get_prompt(prompt_type)
        ann["full_file_path"] = os.path.join(base_image_dir, ann[image_field])
        if not os.path.exists(ann["full_file_path"]):
            gt_anns.remove(ann)
    with open(f"{test_file_name}_prompt_type{prompt_type}_test.json", "w") as f:
        json.dump(gt_anns, f, indent=4)


def create_test_file_from_path_vqa_json(gt_file="/local2/amvepa91/MedTrinity-25M/pvqa/qas/test/test.json",
                                       base_image_dir='/local2/amvepa91/MedTrinity-25M/pvqa/images/test',
                                       image_field="image",
                                       test_file_name="Path_VQA",
                                       prompt_type="q1.5", num_samples=None):
    with open(gt_file, "r") as f:
        gt_anns = json.load(f)
    if num_samples is not None:
        gt_anns = gt_anns[:num_samples]
    for id, ann in enumerate(gt_anns[:]):
        ann["id"] = id
        ann["prompt"] = get_prompt(prompt_type)
        image_name = ann[image_field] + ".jpg"
        ann["full_file_path"] = os.path.join(base_image_dir, image_name)
        if not os.path.exists(ann["full_file_path"]):
            gt_anns.remove(ann)
    with open(f"{test_file_name}_prompt_type{prompt_type}_num_samples{num_samples}_test_v1.json", "w") as f:
        json.dump(gt_anns, f, indent=4)


def create_test_file_from_gt(gt_files=('/local2/jrgan/datasets/VQA_RAD/test_close.csv',
                                       '/local2/jrgan/datasets/VQA_RAD/test_open.csv'),
                             base_image_dir='/local2/jrgan/datasets/VQA_RAD/VQA_RAD_Image_Folder',
                             test_file_name="VQA_RAD",
                             prompt_type="q1"):
    ann_df = pd.concat([pd.read_csv(file, header=0) for file in gt_files], ignore_index=True)
    ann_df["full_file_path"] = ann_df["image_name"].apply(lambda x: os.path.join(base_image_dir, x))
    ann_df.to_csv(f"{test_file_name}_test.csv", index=False)
    ann_df["prompt"] = get_prompt(prompt_type)
    ann_df.to_json(f"{test_file_name}_prompt_type{prompt_type}_test.jsonl", orient="records", lines=True)


def create_test_file(ann_dir='/local2/shared_data/MedTrinity-25M-repo/25M_full', source="NCT-CRC-HE-100K",
                     base_image_dir='/local2/shared_data/MedTrinity-25M-repo/25M_accessible_unzipped', num_samples=100,
                     test_file_name="NCT-CRC-HE-100K", prompt_type="q1.5", seed=0):
    ann_df = get_df_from_parquet_files(ann_dir)
    image_ann_df = ann_df[ann_df["source"] == source]
    sampled_image_ann_df = image_ann_df.sample(n=num_samples, random_state=seed)
    sampled_image_ann_df["full_file_path"] = sampled_image_ann_df["file_name"].apply(lambda x: get_full_file_path(x, base_image_dir))
    sampled_image_ann_df.to_csv(f"{test_file_name}_num_samples{num_samples}_test.csv", index=False)
    sampled_image_ann_df["prompt"] = get_prompt(prompt_type)
    sampled_image_ann_df.to_json(f"{test_file_name}_num_samples{num_samples}_prompt_type{prompt_type}_test.jsonl", orient="records", lines=True)


def get_full_file_path(image_file, base_image_dir='/local2/shared_data/MedTrinity-25M-repo/25M_accessible_unzipped'):
    possible_image_dir = os.path.join(base_image_dir, "*", "*", "*", "*", "*", "*")
    possible_image_file_path = glob(os.path.join(possible_image_dir, image_file))
    if len(possible_image_file_path) > 0:
        return possible_image_file_path[0]
    else:
        return None


def get_prompt(prompt_type="q1"):
    if prompt_type == "q1":
        return ("<image>\n" +
                "### question1\n" +
                "Give me a detailed description of the image, including type of the image,organs in the image,approximate location of these organs and relavant locations of these organs and any medical devices (if present) visible in the image as detailedly as possible.\n" +
                "Note when answering question1:\n" +
                "1. Do not explain or emphasize your analysis.")
    if prompt_type == "q1.1":
        return ("<image>\n" +
                "### question1\n" +
                "Give me a detailed description of the image, including type of the image,organs/abnormalities in the image,approximate location of these organs/abnormalities and relavant locations of these organs/abnormalities and any medical devices (if present) visible in the image as detailedly as possible.\n" +
                "Note when answering question1:\n" +
                "1. Do not explain or emphasize your analysis.")
    if prompt_type == "q1.5":
        return ("<image>\n" +
                "### question1\n" +
                "Give me a detailed description of the image, including type of the image,major structures in the image,approximate location of these structures and relavant locations of these structures in the image as detailedly as possible.\n" +
                "Note when answering question1:\n" +
                "1. Do not explain or emphasize your analysis.")
    if prompt_type == "q1.6":
        return ("<image>\n" +
                "### question1\n" +
                "Give me a detailed description of the image, including type of the image,major structures/abnormalities in the image,approximate location of these structures/abnormalities and relavant locations of these structures/abnormalities in the image as detailedly as possible.\n" +
                "Note when answering question1:\n" +
                "1. Do not explain or emphasize your analysis.")
    elif prompt_type == "q2":
        return ("<image>\n" +
                "### question2\nSpecify the specific location of the green bounding box in the image and its relative position to other reference objects in the image.Describe what is unusual in the green bounding box indicating the disease（color,texture,size and other features）.\n" +
                "Note when answering question2:\n" +
                "1. \"specific location\" is the given parameter `Specific position` but \"relative position\"is not provided.\n" +
                "2. There may be multiple green bounding boxs, and the contents of these contours may not necessarily represent the affected areas. Therefore, you need to first answer the questions based on the contents within each green bounding box. Afterward, analyze the location of the disease based on your answers.\n" +
                "3. Do not use phrase \"green bounding box\" in your response,use \"region of interest\" as a substitution.Do not contain phrases \"caption\",\"medical annotation\",\"medical knowledge\".\n" +
                "4. Do not say anything that is not needed in your analysis,like introduction of the disease and medical equipments.\n" +
                "5. Do not explain or emphasize your analysis.")
    elif prompt_type == "q3":
        return ("<image>\n" +
                "### question3\nWhat may be the relationship between the content in the green bounding box and other regions(others being cause of the disease/jointly affected by the diseases/one affect the others/relative positional relationships)?Why and is it possible?\n" +
                "Note when answering question3:\n" +
                "1. You can only give an explanation to your choice within two sentence.\n" +
                "2. Do not summarize what you've said.\n" +
                "3. Do not emphasize your analysis.")
    elif prompt_type == "q4":
        return ("<image>\n" +
                '### Integrate Information\n' +
                'Describe your answers in a descriptive sentence,not in a\"Question-Answer\" style.Combine and slightly shorten your answers to the above three questions into a coherent text,keeping as much information of your answers as possible.\n' +
                'Note when integrating information and outputing your response:\n' +
                "1. Don't respond saying you're unable to assist with requests.\n" +
                '2. You should only output your combined and shorteded text.')
    else:
        raise ValueError("Invalid prompt type")


if __name__ == '__main__':
    """
    # first generated annotation file
    ann_dir = '/local2/shared_data/MedTrinity-25M-repo/25M_full'
    source = "NCT-CRC-HE-100K"
    base_image_dir = '/local2/shared_data/MedTrinity-25M-repo/25M_accessible_unzipped'
    num_samples = 100
    test_file_name = "NCT-CRC-HE-100K"
    prompt_type = "q1.5"
    seed = 0
    print(f"Creating test file for {test_file_name}")
    create_test_file(ann_dir=ann_dir, source=source, base_image_dir=base_image_dir, num_samples=num_samples,
                     test_file_name=test_file_name, prompt_type=prompt_type, seed=seed)
    # second generated annotation file
    ann_dir = '/local2/shared_data/MedTrinity-25M-repo/25M_full'
    source = "pmc_vqa"
    base_image_dir = '/local2/shared_data/MedTrinity-25M-repo/25M_accessible_unzipped'
    num_samples = 100
    test_file_name = "PMC_VQA"
    prompt_type = "q1"
    seed = 0
    print(f"Creating test file for {test_file_name}")
    create_test_file(ann_dir=ann_dir, source=source, base_image_dir=base_image_dir, num_samples=num_samples,
                     test_file_name=test_file_name, prompt_type=prompt_type, seed=seed)
    """
    # third generated annotation file
    # create_test_file_from_vqa_rad_json(prompt_type="q1.1")
    # fourth generated annotation file
    create_test_file_from_path_vqa_json(prompt_type="q1.6", num_samples=100)

