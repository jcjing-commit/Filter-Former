"""Compute evaluation metrics for a single experiment."""

__author__ = "Paul Bergmann, David Sattlegger"
__copyright__ = "2021, MVTec Software GmbH"

import json
from os import listdir, makedirs, path

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import generic_util as util
# from pro_curve_util import compute_pro
# from roc_curve_util import compute_classification_roc
from scipy.ndimage.measurements import label
from skimage.transform import resize
from sklearn.metrics import roc_auc_score,  precision_recall_curve, average_precision_score


OBJECT_NAMES2 = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal',
                'vial', 'wallplugs', 'walnuts']

OBJECT_NAMES2 = ['rice', 'walnuts']



def parse_user_arguments():
    """Parse user arguments for the evaluation of a method on the MVTec AD
    dataset.

    Returns:
        Parsed user arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--anomaly_maps_dir',
                        # required=True,
                        default='./workout/anomaly_images/' ,        #change
                        help="""Path to the directory that contains the anomaly
                                maps of the evaluated method.""")

    parser.add_argument('--dataset_base_dir',
                        # required=True,
                        default='../mvtec_ad_2/',
                        help="""Path to the directory that contains the dataset
                                images of the MVTec AD dataset.""")

    parser.add_argument('--output_dir',
                        default='./workout' ,                 #change
                        help="""Path to the directory to store evaluation
                                results. If no output directory is specified,
                                the results are not written to drive.""")

    parser.add_argument('--pro_integration_limit',
                        type=float,
                        default=0.1,
                        help="""Integration limit to compute the area under
                                the PRO curve. Must lie within the interval
                                of (0.0, 1.0].""")

    parser.add_argument('--evaluated_objects',
                        nargs='+',
                        default=OBJECT_NAMES2, #['vial'],                                    #change 
                        help="""List of objects to be evaluated. By default,
                                all dataset objects will be evaluated.""",
                    )
                        # choices=util.OBJECT_NAMES,)
                        # default=util.OBJECT_NAMES)

    args = parser.parse_args()

    # Check that the PRO integration limit is within the valid range.
    assert 0.0 < args.pro_integration_limit <= 1.0

    return args



def compute_pro(anomaly_maps, ground_truth_maps):
    """Compute the PRO curve for a set of anomaly maps with corresponding ground
    truth maps.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a
          real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
          contain binary-valued ground truth labels for each pixel.
          0 indicates that a pixel is anomaly-free.
          1 indicates that a pixel contains an anomaly.

    Returns:
        fprs: numpy array of false positive rates.
        pros: numpy array of corresponding PRO values.
    """

    print("Compute PRO curve...")

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        # Compute the mask that gives us all ok pixels.
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        # added to the set of positives.
        # fp_change needs to be normalized later when we know the final value
        # of num_ok_pixels -> right now it is only the change in the number of
        # false positives
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        # added to the set of positives.
        # pro_change needs to be normalized later when we know the final value
        # of num_gt_regions.
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores.
    print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    # b=a[ind] showed to be more memory efficient.
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values.
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    # Merge (FPR, PRO) points that occur together at the same threshold.
    # For those points, only the final (FPR, PRO) point should be kept.
    # That is because that point is the one that takes all changes
    # to the FPR and the PRO at the respective threshold into account.
    # -> keep_mask is True if the subsequent score is different from the
    # score at the respective position.
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    anomaly_scores_sorted = anomaly_scores_sorted[keep_mask]
    # del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    # To mitigate the adding up of numerical errors during the np.cumsum calls,
    # make sure that the curve ends at (1, 1) and does not contain values > 1.
    np.clip(fprs, a_min=None, a_max=1., out=fprs)
    np.clip(pros, a_min=None, a_max=1., out=pros)

    # Make the fprs and pros start at 0 and end at 1.
    zero = np.array([0.])
    one = np.array([1.])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one)), anomaly_scores_sorted


def calculate_optimal_threshold(anomaly_maps, ground_truth_maps):
    fprs, pros, anomaly_scores_sorted = compute_pro(anomaly_maps, ground_truth_maps)
    
    # 计算F1分数
    f1_scores = 2 * (1 - fprs) * pros / ((1 - fprs) + pros)
    
    # 找到F1分数最大的点
    optimal_index = np.argmax(f1_scores)
    optimal_threshold = anomaly_scores_sorted[optimal_index]
    
    return optimal_threshold, f1_scores[optimal_index]


def parse_dataset_files(object_name, dataset_base_dir, anomaly_maps_dir):
    """Parse the filenames for one object of the MVTec AD dataset.

    Args:
        object_name: Name of the dataset object.
        dataset_base_dir: Base directory of the MVTec AD dataset.
        anomaly_maps_dir: Base directory where anomaly maps are located.
    """
    # assert object_name in util.OBJECT_NAMES

    # Store a list of all ground truth filenames.
    gt_filenames = []

    # Store a list of all corresponding anomaly map filenames.
    prediction_filenames = []

    # Test images are located here.
    test_dir = path.join(dataset_base_dir, object_name, 'test_public')
    # gt_base_dir = path.join(dataset_base_dir, object_name, 'ground_truth')
    gt_base_dir = path.join(test_dir, 'ground_truth')
    anomaly_maps_base_dir = path.join(anomaly_maps_dir, object_name, 'test_public')

    # List all ground truth and corresponding anomaly images.
    for subdir in listdir(str(test_dir)):

        if not subdir.replace('_', '').isalpha():
            continue

        # Get paths to all test images in the dataset for this subdir.
        test_images = [path.splitext(file)[0]
                       for file
                       in listdir(path.join(test_dir, subdir))
                       if path.splitext(file)[1] == '.png']

        # If subdir is not 'good', derive corresponding GT names.
        if subdir != 'good':
            gt_filenames.extend(
                [path.join(gt_base_dir, subdir, file + '_mask.png')
                 for file in test_images])
        else:
            # No ground truth maps exist for anomaly-free images.
            gt_filenames.extend([None] * len(test_images))

        # Fetch corresponding anomaly maps.
        prediction_filenames.extend(
            [path.join(anomaly_maps_base_dir, subdir, file)
             for file in test_images])

    print(f"Parsed {len(gt_filenames)} ground truth image files.")

    return gt_filenames, prediction_filenames


def calculate_au_pro_au_roc(gt_filenames,
                            prediction_filenames,
                            integration_limit):
    """Compute the area under the PRO curve for a set of ground truth images
    and corresponding anomaly images.

    In addition, the function computes the area under the ROC curve for image
    level classification.

    Args:
        gt_filenames: List of filenames that contain the ground truth images
          for a single dataset object.
        prediction_filenames: List of filenames that contain the corresponding
          anomaly images for each ground truth image.
        integration_limit: Integration limit to use when computing the area
          under the PRO curve.

    Returns:
        au_pro: Area under the PRO curve computed up to the given integration
          limit.
        au_roc: Area under the ROC curve.
        pro_curve: PRO curve values for localization (fpr,pro).
        roc_curve: ROC curve values for image level classifiction (fpr,tpr).
    """
    # Read all ground truth and anomaly images.
    ground_truth = []
    predictions = []

    gt_filenames = np.array(gt_filenames)
    gt_path = gt_filenames[gt_filenames!=None][0]
    gt_img = np.asarray(Image.open(gt_path))
    gt_shape = gt_img.shape

    print("Read ground truth files and corresponding predictions...")
    for (gt_name, pred_name) in tqdm(zip(gt_filenames, prediction_filenames),
                                     total=len(gt_filenames)):
        prediction = util.read_tiff(pred_name)
        # 使用双线性插值进行缩放
        prediction = resize(prediction, gt_shape, order=1, mode='reflect', anti_aliasing=True)
        predictions.append(prediction)

        if gt_name is not None:
            # ground_truth.append(np.asarray(Image.open(gt_name)))
            ground_truth_img = np.asarray(Image.open(gt_name))
            # 将所有非零值设置为1
            ground_truth_img = ground_truth_img.copy()
            ground_truth_img[ground_truth_img != 0] = 1
            ground_truth.append(ground_truth_img)
        else:
            ground_truth.append(np.zeros(prediction.shape))

    # Compute the PRO curve
    

    # optimal_threshold,f1_scores = calculate_optimal_threshold(predictions, ground_truth)
    anomaly_scores_flat = np.array(predictions).ravel()
    y_test = np.array(ground_truth).ravel()
    precs, recs, thrs = precision_recall_curve(y_test, anomaly_scores_flat)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    max_f1 = np.max(f1s)


    print("最优阈值是:", best_thr)
    print("F1-scores:", max_f1)
    return best_thr, max_f1
    # # Compute the area under the PRO curve.
    # au_pro = util.trapezoid(
    #     pro_curve[0], pro_curve[1], x_max=integration_limit)
    # au_pro /= integration_limit
    # print(f"AU-PRO (FPR limit: {integration_limit}): {au_pro}")

    # # Derive binary labels for each input image:
    # # (0 = anomaly free, 1 = anomalous).
    # binary_labels = [int(np.any(x > 0)) for x in ground_truth]
    # del ground_truth

    # # Compute the classification ROC curve.
    # roc_curve = compute_classification_roc(
    #     anomaly_maps=predictions,
    #     scoring_function=np.max,
    #     ground_truth_labels=binary_labels)

    # # Compute the area under the classification ROC curve.
    # au_roc = util.trapezoid(roc_curve[0], roc_curve[1])
    # print(f"Image-level classification AU-ROC: {au_roc}")

    # # Return the evaluation metrics.
    # return au_pro, au_roc, pro_curve, roc_curve


def convert_to_builtin_type(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(item) for item in obj]
    else:
        return obj
    
def main():
    """Calculate the performance metrics for a single experiment on the
    MVTec AD dataset.
    """
    # Parse user arguments.
    args = parse_user_arguments()

    if not args.dataset_base_dir.endswith('mvtec_ad_2'):
        args.dataset_base_dir = path.join(args.dataset_base_dir, 'mvtec_ad_2')

    # Store evaluation results in this dictionary.
    evaluation_dict = dict()

    # Keep track of the mean performance measures.
    au_pros = []
    au_rocs = []

    # Evaluate each dataset object separately.
    for obj in args.evaluated_objects:
        print(f"=== threshold evaluation for object {obj} ===")
        evaluation_dict[obj] = dict()

        # Parse the filenames of all ground truth and corresponding anomaly
        # images for this object.
        gt_filenames, prediction_filenames = \
            parse_dataset_files(
                object_name=obj,
                dataset_base_dir=args.dataset_base_dir,
                anomaly_maps_dir=args.anomaly_maps_dir)

        # Calculate the PRO and ROC curves.
        optimal_threshold, f1_scores = \
            calculate_au_pro_au_roc(
                gt_filenames,
                prediction_filenames,
                args.pro_integration_limit)

        evaluation_dict[obj]['optimal_threshold'] = optimal_threshold
        evaluation_dict[obj]['f1_scores'] = f1_scores


        print('\n')

    # Compute the mean of the performance measures.


    # If required, write evaluation metrics to drive.
    if args.output_dir is not None:
        makedirs(args.output_dir, exist_ok=True)
        evaluation_dict = convert_to_builtin_type(evaluation_dict)
        with open(path.join(args.output_dir, 'threshold_value.json'), 'w') as file:
            json.dump(evaluation_dict, file, indent=4)

        print(f"Wrote metrics to {path.join(args.output_dir, 'threshold_value.json')}")


if __name__ == "__main__":
    main()
