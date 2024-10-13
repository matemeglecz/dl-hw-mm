import glob 
import os
import shutil
import numpy as np
import json
import pydicom
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import scipy.interpolate
import re

def check_datset(root):
    """
    Check for missing contours, dicom files, subfolders, etc.

    Args:
        root (str, path): root folder of the dataset, whith Patient (...) subfolders
    """
    required_slices = [
        "T1_map_apex",
        "T1_map_base",
        "T1_map_mid_",
        "T1_Mapping_",
        "T2_map_apex",
        "T2_map_base",
        "T2_map_mid_",
        "T2_Mapping_"]
    allowed_slices = required_slices + [
        "T1_map_apex_uncorrected",
        "T1_map_apex_contrast",
        "T1_map_base_uncorrected",
        "T1_map_base_contrast",
        "T1_map_mid_uncorrected",
        "T1_map_mid_contrast",
        "T2_map_apex_uncorrected",
        "T2_map_base_uncorrected",
        "T2_map_mid_uncorrected"]

    patients = sorted(glob.glob(os.path.join(root,'*')))
    
    for patient_folder in patients:
        print(os.path.basename(patient_folder))

        slice_paths = glob.glob(os.path.join(patient_folder,'*'))
        slice_folders = [os.path.basename(slice_folder) for slice_folder in slice_paths]

        # Check if all required slices/subfolders are found for the patient
        for required_slice in required_slices:
            if required_slice not in slice_folders:
                raise FileNotFoundError(f"Slice folder {required_slice} missing from {os.path.basename(patient_folder)}")
        
        # Check for unexpected slices/folders
        # If a folder is allowed, pop it from the list. 
        # If folders remain in the list after the loop, they are not allowed.     
        for allowed_slice in allowed_slices:
            if allowed_slice in slice_folders:
                slice_folders.remove(allowed_slice)
                continue
        if slice_folders != []:
            print(f"  Unexpected subfolders found in {os.path.basename(patient_folder)}: {slice_folders}")
        
        for slice_path in slice_paths:
            # Check for empty slice folders and skip checking their content
            if os.listdir(slice_path) == []:
                continue
            # Skip _uncorrected folders from checking their contents
            if "_uncorrected" in os.path.basename(slice_path):
                continue
            # Skip _contrast folders from checking their contents, because these don't have any contours
            if "_contrast" in os.path.basename(slice_path):
                continue
            # Check if contours are found in ..._map_... folders
            if "_map_" in os.path.basename(slice_path):
                expected_contours_path = os.path.join(slice_path, "Contours.json")
                if not os.path.exists(expected_contours_path):
                    print(f"  Contours file missing: {expected_contours_path.replace(root+'/','')}")
            # Check for missing mapping files
            elif  "Mapping" in os.path.basename(slice_path):
                for mapping_dcm_path in glob.glob(os.path.join(slice_path, '*.dcm')):
                    # Mapping dcms should be named Apex.dcm, Base.dcm, Mid.dcm, Apex_2.dcm, Base_2.dcm, Mid_2.dcm
                    # Contours for _2 files are not loaded, but their presence is not reported as an error!
                    mapping_dcm_type = os.path.basename(mapping_dcm_path).replace("_2", "")
                    if mapping_dcm_type not in ["Apex.dcm", "Base.dcm", "Mid.dcm"]:
                        print(f"  Unexpected Mapping file: {mapping_dcm_path.replace(root+'/','')}")

                

def construct_samples_list(root, contours_filename="Contours.json"):
    """
    Search for image files and matching segmentation contour files.
    Construct a list of [[image path, segmentation contour file path], ...] pairs for each image. 
    The same contour file is used for multiple images.
    Only inlcudes images with valid contour files. Don't use this if you only need images. 

    Dicom path                            Contour path
    ------------------------------------  -------------------------------------
    Patient (1)/T1_map_mid_/...23102.dcm  Patient (1)/T1_map_mid_/Contours.json
    Patient (1)/T1_map_mid_/...23097.dcm  Patient (1)/T1_map_mid_/Contours.json
    Patient (1)/T1_map_mid_/...23100.dcm  Patient (1)/T1_map_mid_/Contours.json
    Patient (1)/T1_map_mid_/...23099.dcm  Patient (1)/T1_map_mid_/Contours.json
    Patient (1)/T1_map_mid_/...23101.dcm  Patient (1)/T1_map_mid_/Contours.json

    Args:
        root (str): dataset root folder
        contours_filename (str) : Filename for contour files, intended for use in the inter-observer analysis

    Returns:
        list of tuples (str, str): List of tuples, each with the following elements: dicon_path, contour_path
    """
    samples = []
    # No warnings for missing contours in this function. Simply skip incorrect samples. Run check_dataset to check for missing data.
    for patient_folder in sorted(glob.glob(os.path.join(root, '*', 'Patient (*'))):
        for slice_path in glob.glob(os.path.join(patient_folder,'*')):
            if "_uncorrected" in os.path.basename(slice_path):
                continue
            elif "_map_" in os.path.basename(slice_path):
                expected_contours_path = os.path.join(slice_path, contours_filename)
                if os.path.exists(expected_contours_path):
                    dcm_paths =  glob.glob(os.path.join(slice_path,'*.dcm'))
                    contour_paths = [expected_contours_path]*len(dcm_paths)
                    samples.extend(zip(dcm_paths, contour_paths))
            elif  "Mapping" in os.path.basename(slice_path):
                if "T1_Mapping" in os.path.basename(slice_path):
                    contours_folders = {"Apex.dcm": "T1_map_apex", 
                                        "Base.dcm": "T1_map_base", 
                                        "Mid.dcm": "T1_map_mid_"}
                elif "T2_Mapping" in os.path.basename(slice_path):
                    contours_folders = {"Apex.dcm": "T2_map_apex", 
                                        "Base.dcm": "T2_map_base", 
                                        "Mid.dcm": "T2_map_mid_"} 
                for mapping_dcm_path in glob.glob(os.path.join(slice_path, '*.dcm')):
                    # Mapping dcms should be named Apex.dcm, Base.dcm, Mid.dcm, Apex_2.dcm, Base_2.dcm, Mid_2.dcm
                    # Contours for _2 files are not loaded!
                    mapping_dcm_type = os.path.basename(mapping_dcm_path)
                    if mapping_dcm_type not in ["Apex.dcm", "Base.dcm", "Mid.dcm"]:
                        continue
                    conotours_folder = contours_folders[mapping_dcm_type]
                    expected_contours_path = os.path.join(patient_folder, conotours_folder, contours_filename)
                    if os.path.exists(expected_contours_path): 
                        samples.append((mapping_dcm_path, expected_contours_path))

    return samples

def split_samples_list(samples, split_patient_ids):
    """
    Select a subset of the samples list based on a list of patient ids. 

    Args:
        samples (list of [image path, segmentation contour file path] tuples): as returned by construct_samples_list
        split_patient_ids (list of ints): list of patient ids

    Returns:
        (list of [image path, segmentation contour file path] tuples): samples of patients whose id is in the split_patient_ids list
    """
    split_samples = []
    for sample in samples:
        if isinstance(sample, tuple) or isinstance(sample, list):
            path_parts = sample[0].split(os.path.sep)
        else:
            path_parts = sample.split(os.path.sep) # for the unlabeled sample list, wich is a list containing paths only
        # e.g. path_parts: Patient (1)/T1_map_base/Contours.json
        patient_folder = path_parts[-3]   
        # e.g. patient_folder: Patient (1)
        patient_id = int(patient_folder[patient_folder.index('(') + 1:patient_folder.index(')')])
        if patient_id in split_patient_ids:
            split_samples.append(sample)
    return split_samples


def print_diagnostics(root, samples, list_shapes=False, get_mean_std=False):
    contours_list = np.unique([sample[1] for sample in samples])
    print(f"Found {len(samples)} dcm files with contours and {len(contours_list)} contours files.")
    all_dcms =  glob.glob(os.path.join(root, "**/*.dcm"), recursive=True)
    uncorrected_dicoms = [dcm_path for dcm_path in all_dcms if "_uncorrected" in dcm_path]
    print(f"Found {len(uncorrected_dicoms)} uncorrected dcm files.")
    contrast_dicoms = [dcm_path for dcm_path in all_dcms if "_contrast" in dcm_path or "_2.dcm" in dcm_path]
    print(f"Found {len(contrast_dicoms)} contrast dcm files.")
    dcms_with_missing_contours = len(all_dcms) - len(samples) - len(uncorrected_dicoms)  - len(contrast_dicoms)
    print(f"Couldn't identify contours file for  {dcms_with_missing_contours} dcm files (not including uncorrected or contrast files).")
    
    if list_shapes or get_mean_std:
        shapes = []
        all_pixels = []
        # for dicom_path in tqdm(sorted(glob.glob(os.path.join(root,'**', '*.dcm'), recursive=True)), desc="Loading all dicoms for dataset stats"):
        for dicom_path in tqdm(sorted([sample[0] for sample in samples]), desc="Loading all dicoms for dataset stats"):
            try:
                image = pydicom.dcmread(dicom_path).pixel_array # force True
            except:
                #print(e)
                print("Incorrect dicom:", dicom_path)
                f = open("bad_mris.txt", "a")
                f.write(dicom_path + '\n')
                f.close()
                continue
            if image.shape not in shapes:
                shapes.append(image.shape)
            if get_mean_std:
                all_pixels.append(image.flatten())
        if list_shapes:
            print("Found dicom images with shapes:\n  ", sorted(shapes))
        if get_mean_std:
            all_pixels = np.concatenate(all_pixels)
            print(f"Pixel mean for all images {all_pixels.mean()}")
            print(f"Pixel std for all images {all_pixels.std()}")
        
        stats = {"shapes": shapes,
                 "mean": all_pixels.mean(), 
                 "std": all_pixels.std(),
                 "all_pixels": all_pixels}
        return stats


def rename_Mapping_2_mapping(root: str):
    """ DON'T USE THIS
    First 68 patients has Mapping folder with capital M this function renames these folder to use lower case m

    >>> from data.mapping_utils import rename_Mapping_2_mapping
    >>> rename_Mapping_2_mapping("/home1/ssl-phd/data/mapping")
    
    Args:
        root (str): dataset root folder with Patient (xx) folders in it.
    """
    for patient_folder in sorted(glob.glob(os.path.join(root,'*'))):
        for slice_path in glob.glob(os.path.join(patient_folder,'*')):
            if "T1_Mapping_"  == os.path.basename(slice_path):
                new_name = os.path.join(os.path.dirname(slice_path), "T1_mapping_")
                print(f"{slice_path} -> {new_name}")
                shutil.move(slice_path, new_name)
            elif "T2_Mapping_" == os.path.basename(slice_path):
                new_name = os.path.join(os.path.dirname(slice_path), "T2_mapping_")
                print(f"{slice_path} -> {new_name}")
                shutil.move(slice_path, new_name)


def rename_mapping_2_Mapping(root: str, dry_run=True):
    """Initially we received some patients, who has Mapping folder with capital M, some with lower case m.
    This function renames folders consistently to capital M Mapping_

    >>> from data.mapping_utils import rename_mapping_2_Mapping
    >>> rename_mapping_2_Mapping("/mnt/hdd2/se", dry_run=False)
    
    Args:
        root (str): dataset root folder with Patient (xx) folders in it or in it's subfolders (1 level deep).
        dry_run (bool): If True only prints what 
    """
    for patient_folder in sorted(glob.glob(os.path.join(root,'**/Patient (*)'), recursive=True)):
        for slice_path in glob.glob(os.path.join(patient_folder,'*')):
            if "T1_mapping_"  == os.path.basename(slice_path):
                new_name = os.path.join(os.path.dirname(slice_path), "T1_Mapping_")
                print(f"{slice_path} -> {new_name}")
                if not dry_run:
                    shutil.move(slice_path, new_name)
            elif "T2_mapping_" == os.path.basename(slice_path):
                new_name = os.path.join(os.path.dirname(slice_path), "T2_Mapping_")
                print(f"{slice_path} -> {new_name}")
                if not dry_run:
                    shutil.move(slice_path, new_name)
    if dry_run: 
        print("Dry run completed. Listed all folders to be renamed, but didn't perform renaming.")


def get_dataset_mean_std(root: str):
    """
    Args:
        root (str): dataset root folder

    Returns:
        (float, float): mean, std
    """
    all_pixels = []
    for dcm_path in glob.glob(os.path.join(root, "**/*.dcm"), recursive=True):
        image = pydicom.dcmread(dcm_path).pixel_array
        all_pixels.append(image.flatten())
    all_pixels = np.concatenate(all_pixels)
    return all_pixels.mean(), all_pixels.std()


def load_dicom(path, mode='channels_first', use_modality_lut=True):
    """
    Loads dicom files to a numpy array. 
    Args:
        path (str): 
        mode (str, optional): 'channels_first', 'channels_last', '2d' or None. Defaults to 'channels_first'.
        use_modality_lut (bool, optional): If True applies modality lut to the image. Defaults to True. Using modality lut is the correct way to load dicom images, but it was not used when we trained our models!

    Returns:
        np.ndarray: image data from the dicom file
    """
    try:
        dcm = pydicom.dcmread(path)
    except pydicom.errors.InvalidDicomError as e:
        print(e)
        print("Incorrect dicom:", path)
        dcm = pydicom.dcmread(path, force=True)
        
    try:
        image = dcm.pixel_array
    except AttributeError as e:
        print(e)
        print("Incorrect dicom:", path)
        return None

    if use_modality_lut:
        image = pydicom.pixel_data_handlers.util.apply_modality_lut(image, dcm).astype(np.float32)

    if mode == 'channels_first':
        image = np.expand_dims(image, 0)
    elif mode == 'channels_last': 
        image = np.expand_dims(image, -1)
    else:
        assert mode == '2d' or mode == None, f"Unrecognized loading mode: {mode}!" \
            "Allowed values \'channels_first\', \'channels_last\', \'2d\' or None"
  
    return image


def load_contours(labelfile):
    epicardial_contour = None
    endocardial_contour = None
    try:
        with open(labelfile) as f:
            contour_data = json.load(f)
            epicardial_contour = np.stack([contour_data['epicardial_contours_x'],
                                            contour_data['epicardial_contours_y']], 
                                        axis=1)
            endocardial_contour = np.stack([contour_data['endocardial_contours_x'],
                                            contour_data['endocardial_contours_y']], 
                                            axis=1)
    except ValueError as e:
        print(f"ERROR loading {labelfile}:\n{e}")
    # print("epicardial_contour.shape = ", epicardial_contour.shape)
    # print("endocardial_contour.shape = ", endocardial_contour.shape)
    return epicardial_contour, endocardial_contour


def contours_to_masks(contours, shape, contour_fp_precision_bits = 8, oversampling_factor=4):
    """Converts segmentation contours (epicardial_contour, endocardial_contour) to segmentation mask.
       Region between the two contours is foreground, rest is background. 

    Args:
        contours ([np.ndarray, np.ndarray]): Coordinates of epicardial_contour, endocardial_contour points as ndarrays.
        shape (tuple): Shape of the output mask, shape of the input image
        contour_fp_precision_bits (int, optional): OpenCV's fillPoly accepts contours with fix point representation.
                                                   contour_fp_precision_bits determines the number of fractional bits.
                                                   Contours are rounded to this precision. Defaults to 10.
        oversampling_factor (int, optional): Degree of oversampling to be applied to the input contours before creating the segmentation mask.
                                             Higher values result in more accurate but slower segmentation masks. 
                                             A value of 1 means no oversampling is applied.
                                             Defaults to 4.
    Returns:
        np.ndarray: Segmentation mask
    """
    upscaled_contours = [c*oversampling_factor for c in contours]
    
    # OpenCV's fillPoly accepts contours with fix point representation, passed as int32,
    # whose last 'shift' bits are interpreted as fractional bits.
    # Here we multiply by 2^contour_fp_precision_bits to achieve this representation.
    rounded_conoturs = [np.around(contour*2**contour_fp_precision_bits).astype(np.int32) for contour in upscaled_contours]

    # Rounded contours might have repeated points which could break filling betweeng the two contours.
    rounded_conoturs = [unique_consecutive(contour) for contour in rounded_conoturs]
    
    if len(shape) == 2:
        mask = np.zeros(np.array(shape)*oversampling_factor)
    else:
        mask = np.zeros(np.array(shape[-2:])*oversampling_factor)
    
    # Draw epicardial
    cv2.fillPoly(mask, pts=[rounded_conoturs[0]], color=(1,1,1), shift=contour_fp_precision_bits)

    # Draw endocardial
    tmp_mask = np.zeros_like(mask)
    cv2.fillPoly(tmp_mask, pts=[rounded_conoturs[1]], color=(2,2,2), shift=contour_fp_precision_bits)
    # Erode left ventricular region by 1 pixel, so that the endocardial contour is also inside the myocardial mask
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    cv2.erode(tmp_mask, kernel=kernel, dst=tmp_mask, iterations=1)
    mask[tmp_mask==2] = 2

    # Downscale to original size
    mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask #/ (max(mask.flatten()) + 1)

def find_last_in_row(row, value):
    for i in range(len(row)-1, -1, -1):
        if row[i] == value:
            return i
    return -1

def merge_contours_on_image_from_mask(image, mask, num_classes):
    # check if the last dimension is 3
    if image.shape[2] != 3:
        raise ValueError("The image should have 3 channels")

    # if the smallest value in the mask is 3, then we substract 3 form the values
    if num_classes == 7:
        mask -= 1
        
    if 3 in np.unique(mask):
        mask -= 3
    
    print(np.unique(mask))

    shape = image.shape

    first_contour_found_outer = False
    last_contour_found_outer = False
    first_contour_found_inner = False
    last_contour_found_inner = False
    for i in range(shape[0]):
        # find first and last occurence of 1 in the mask
        if not last_contour_found_outer:
            first = np.argmax(mask[i, :] == 1)
            #print(first)
            if first > 0:
                if not first_contour_found_outer:
                    first_contour_found_outer = True
                    # set image pixel value to green where the mask is 1 in that line
                    last = find_last_in_row(mask[i, :], 1)
                    image[i, first:last+1] = [0, 150, 0]
                else:
                    last = find_last_in_row(mask[i, :], 1)
                    image[i, first] = [0, 150, 0]
                    image[i, last] = [0, 150, 0]
            elif first_contour_found_outer:
                last_contour_found_outer = True
                first = np.argmax(mask[i-1, :] == 1)
                last = find_last_in_row(mask[i-1, :], 1)
                image[i-1, first:last+1] = [0, 150, 0]

        # same for inner with 2 pixel value
        if not last_contour_found_inner:
            first = np.argmax(mask[i, :] == 2)
            if first > 0:
                if not first_contour_found_inner:
                    first_contour_found_inner = True
                    last = find_last_in_row(mask[i, :], 2)
                    image[i, first:last+1] = [150, 0, 0]
                else:
                    last = find_last_in_row(mask[i, :], 2)
                    image[i, first] = [150, 0, 0]
                    image[i, last] = [150, 0, 0]
            elif first_contour_found_inner:
                last_contour_found_inner = True
                first = np.argmax(mask[i-1, :] == 2)
                last = find_last_in_row(mask[i-1, :], 2)
                image[i-1, first:last+1] = [150, 0, 0]
        

    return image


def unique_consecutive(contour):
    """Removes repeated consecutive poitns from a contour array, leaving only one occurance of each point."""
    duplicate_pts = []
    for idx in range(contour.shape[0]-1):
        if np.array_equal(contour[idx], contour[idx+1]):
            duplicate_pts.append(idx)
    return np.delete(contour, duplicate_pts, axis=0)

