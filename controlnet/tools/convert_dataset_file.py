import json
from glob import glob
from tqdm import tqdm
from controlnet.tools.training_classes import get_class_stacks


def get_dataset_file(folder):
    imgs_path = folder + "/images"
    file_output = folder + "/dataset_file.json"
    data = []
    for img_path in tqdm(list(sorted(glob(f"{imgs_path}/*.png")))):
        
        label_path = img_path.replace("images", "labels")
        label_train_path = label_path.replace(".png", "_labelTrainIds.png")

        sentence = get_class_stacks(label_train_path)
        data.append({"text":sentence, "image": img_path, "conditioning_image":label_path})
    
    with open(file_output, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')

        
if __name__ == "__main__":
    folder = r"data/gta"
    get_dataset_file(folder)


        