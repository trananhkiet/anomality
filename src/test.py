import contextlib
import gc
import logging
import os
import sys

import click
import numpy as np
import cv2
import torch
import tqdm

import patchcore.common

import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils


LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}

current_dir=os.getcwd()
def create_draw_mask_folder(dir):
    final_mask=os.path.join(dir,'draw_mask','final_mask')
    if not os.path.exists(final_mask):
        os.makedirs(final_mask, exist_ok=True)
    return final_mask

def save_draw_result(current_dir,b_img,score,class_name_gt,class_name,mask_index):
    final_mask=create_draw_mask_folder(current_dir)
    if(class_name=='good'):
        img_text=show_text(b_img,score,class_name_gt,class_name)
        cv2.imwrite(f'{final_mask}/mask_{class_name}_{mask_index}.jpg',img_text)
    if(class_name=='bad'):
        img_text=show_text(b_img,score,class_name_gt,class_name)
        cv2.imwrite(f'{final_mask}/mask_bad_{mask_index}.jpg',img_text)
def reshape_mask(t):
    '''
    input: [tuple],[int]
    output:[tuple] or [int]
    '''
    return t if isinstance(t,tuple) else (t,t,3)
def stack_mask(mask):
    if len(mask.shape)!=2:
        mask=np.squeeze(mask,axis=2)
    return np.stack([mask,mask,mask],axis=2)
def show_text(image,score,class_name_gt,class_name):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (0,30)
    org_pred = (0,60)

    # fontScale
    fontScale = 6/10
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    text=f"Scores:{score} GT:{class_name_gt} Pred:{class_name}"
    img_text=cv2.putText(image,text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    # img_text=cv2.putText(image, f'Pred:{class_name}', org_pred, font, 
    #                fontScale, color, thickness, cv2.LINE_AA)
    return img_text

def resize_and_reshape(image,input_size):
    ''''
    Input
    image:[np.array](h,w,c)
    input_size: [tuple] (h,w)

    Output
    new_image: (h',w',c')
    '''
    image=cv2.resize(image,reshape_mask(input_size))
    return image


       




@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True) #
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--threshold", type=float, default=0.0, show_default=True)
@click.option("--save_segmentation_images" , default=True)
def main(**kwargs):
    pass
@main.result_callback()
def run(methods, results_path, gpu, seed, threshold, save_segmentation_images):
    methods = {key: item for (key, item) in methods}
    print("results_path: ",results_path, '*****************************************')
    print("save segmentation_images: ", save_segmentation_images)
    os.makedirs(results_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    dataloader_iter = dataloader_iter(seed)
    patchcore_iter, n_patchcores = methods["get_patchcore_iter"]
    patchcore_iter = patchcore_iter(device)
    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError(
            "Please ensure that #PatchCores == #Datasets or #PatchCores == 1!"
        )

    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["testing"].name

        with device_context:

            torch.cuda.empty_cache()
            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                
                # This 2 variable to calculate the accuracy
                list_class_name_gt = []
                list_class_name_predict = []
                score_cluster=[]
                mask_index=0
                input_size=(384,384)
                with tqdm.tqdm(dataloaders["testing"], desc="Inferring...", leave=False) as data_iterator:
                    for image_dict in data_iterator:
                        # labels_gt.extend(image["is_anomaly"].numpy().tolist())
                        # masks_gt.extend(image["mask"].numpy().tolist())
                        # image = image["image"]
                        image = image_dict['image']
                        class_name_gt = image_dict['is_anomaly']
                        scores, masks = PatchCore.predict(image)
                        score, mask = scores[0], masks[0]
                        class_name = "bad" if score >= threshold else "good"
                        list_class_name_predict.append(class_name)

                        binary_mask = ((mask - mask.min()) / (mask.max() - mask.min())) * 255
                        # _____________________________________segment defect area_____________________________________#
                        img=cv2.imread(image_dict['image_path'][0])
                        binary=binary_mask.astype(np.float).reshape(binary_mask.shape[0],binary_mask.shape[1])
                       
                        max_val=int(np.max(binary))
                        min_val=int((np.max(binary)+5)/2)
                        _,b_img=cv2.threshold(binary,min_val,max_val,cv2.THRESH_BINARY)
                        # print(b_img.shape,binary.shape,img.shape)
                        concat_img=resize_and_reshape(img,input_size)
                        b_img=resize_and_reshape(b_img,input_size)
                        binary=resize_and_reshape(binary_mask,input_size)
                        # concat_img,b_img,binary=resize_and_reshape(image,b_img,binary,input_size)

                        rbg_mask,rgb_binary=stack_mask(b_img),stack_mask(binary)
                        # print(rbg_mask.shape,rgb_binary.shape,concat_img.shape)
                        final_result=np.concatenate([rbg_mask,rgb_binary,concat_img],axis=1)
                        #_________________________________________________________________________________________________#
                        class_name_gt = "good" if int(class_name_gt.item()) == 0 else "bad"
                        list_class_name_gt.append(class_name_gt)

                        with open("results.txt", "a") as file:
                            file.write(f"score: {score}, ground_truth: {class_name_gt} , predict: {class_name}\n")
                        #_______________________________________Save__________________________________________________________#
                        save_draw_result(current_dir,final_result,score,class_name_gt,class_name,mask_index)
                        mask_index+=1


                # Accuracy
                num_of_correct_cases = sum([1 for x, y in zip(list_class_name_gt, list_class_name_predict) if x == y])
                accuracy = (num_of_correct_cases/len(list_class_name_gt))*100
                print(f"*=*=*=* The classification accuracy of model: {round(accuracy, 1)}%. *=*=*=*")
                
            del PatchCore_list
            gc.collect()

        LOGGER.info("\n\n-----\n")

    # result_metric_names = list(result_collect[-1].keys())[1:]
    # result_dataset_names = [results["dataset_name"] for results in result_collect]
    # result_scores = [list(results.values())[1:] for results in result_collect]
    # patchcore.utils.compute_and_store_final_results(
    #     results_path,
    #     result_scores,
    #     column_names=result_metric_names,
    #     row_names=result_dataset_names,
    # )


@main.command("patch_core_loader")
# Pretraining-specific parameters.
@click.option("--patch_core_paths", "-p", type=str, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_workers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", [get_patchcore_iter, len(patch_core_paths)])


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name, data_path, subdatasets, batch_size, resize, imagesize, num_workers, augment
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST_NO_MASK,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset

            dataloader_dict = {"testing": test_dataloader}

            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
