# This is the class file for the Pseudo Label generation.
# We can call this class to generate the pseudo labels for the unlabeled data.
# Author: Ashiqur Rahman
# URL: https://github.com/ashiqur-rony

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
import skimage.morphology as morphology
from skimage.morphology import closing, remove_small_objects, remove_small_holes
from skimage.segmentation import clear_border
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
from shapely.validation import make_valid
import cv2
import json
import logging

# The Utility file contains the helper functions for the pseudo label generation.
from utility import *

class PseudoLabelAnnotation:
    def __init__(self, mode='single', path=None, threshold_direction=1, tolerance=0.0, **kwargs):
        """Initializes the PseudoLabelAnnotation class.
        :param mode: string, single or batch
        :param path: string, path to the image or directory of images
        :param threshold_direction: int, direction of thresholding (1 for above, -1 for below)
        :param tolerance: float, threshold for the pseudo label generation (default is 0, which means no tolerance)
        :param kwargs: additional arguments for future use
        """
        self.mode = mode
        self.path = path
        self.threshold_direction = threshold_direction
        self.tolerance = tolerance

        self.start_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # See if we want to print the outputs to the console or not, default is False
        self.print = kwargs.get('print', False)

        # See if we want to save the outputs to a file or not, default is False
        self.save = kwargs.get('save', False)
        if self.save:
            self.save_dir = kwargs.get('save_dir', f'pseudo_labels_output{self.start_time}')

        # Set a counter for number of saved images
        self.saved_image_count = 0

        # Set up logging
        logging.basicConfig(
            filename=f'pseudo_label_generation_{self.start_time}.log',
            filemode='a',  # 'a' to append, 'w' to overwrite each time
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

        # Generate log messages
        logging.info("Pseudo label generation started at: %s", self.start_time)
        logging.info("Mode: %s", self.mode)
        logging.info("Path: %s", self.path)

    def generate_pseudo_labels(self):
        """Generates pseudo labels based on the specified mode and path.
        :return: dictionary, mapping of image paths to their corresponding pseudo label segmentations in coco format
        """
        if self.mode == 'single':
            if self.path is None:
                logging.error("No path given for pseudo label generation.")
                raise ValueError("Path to a single image must be provided for pseudo label generation.")

            coco_annotations, width, height = self._process_single_image(self.path)
            annotation_dict = {
                self.path: {
                    'annotations': coco_annotations,
                    'width': width,
                    'height': height
                }
            }
            return annotation_dict
        elif self.mode == 'batch':
            if self.path is None:
                logging.error("No path given for pseudo label generation.")
                raise ValueError("Path to a directory must be provided for pseudo label generation.")

            if self.path not in sys.path:
                logging.info("Adding path to system path: %s", self.path)
                sys.path.append(self.path)

            return self._process_batch_images(self.path)
        else:
            logging.error("Invalid mode for pseudo label generation.")
            raise ValueError("Invalid mode. Choose 'single' or 'batch'.")

    def get_normalized_polygons(self, annotations, desired_width, desired_height, **kwargs):
        """Get the polygons from the annotations and normalize them to the desired width and height. Additionally
        validate the polygons and make them valid if they are not.
        :param annotations: dictionary, raw annotations to be normalized
        :param desired_width: int, desired width for the normalized annotations
        :param desired_height: int, desired height for the normalized annotations
        :param kwargs: additional arguments for future use
        :return: list, list of normalized polygons in desired dimensions
        """

        normalized_polygons = []
        logging.info("Getting normalized polygons...")

        custom_validation = kwargs.get('custom_validation', False)
        ignore_invalid = kwargs.get('ignore_invalid', False)

        for image_path, data in annotations.items():
            logging.info("Processing image: %s", image_path)
            width = data['width']
            height = data['height']
            coco_annotations = data['annotations']

            for ann in coco_annotations:
                segmentation = ann['segmentation']
                if not segmentation or not isinstance(segmentation, list) or not segmentation[0]:
                    logging.info(f"Skipping annotation {ann['id']} with missing or invalid segmentation data.")
                    continue

                for polygon in segmentation:
                    if len(polygon) < 6:  # A valid polygon must have at least 3 points (6 coordinates)
                        logging.info(f"Skipping invalid polygon with less than 3 points: {polygon}")
                        continue

                    # Normalize the polygon coordinates to the image dimensions
                    # We will convert to (desired_width, desired_height) dimensions, so we need to scale the coordinates accordingly
                    normalized_polygon = [(coord / width * desired_width if i % 2 == 0 else coord / height * desired_height) for
                                          i, coord in enumerate(polygon)]
                    normalized_polygons.append(normalized_polygon)

        # Validate the polygons and make them valid if they are not
        valid_polygons = []
        for polygon in normalized_polygons:
            if len(polygon) % 2 != 0:
                logging.warning(f"Invalid polygon with odd number of coordinates: {polygon}")

            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            poly = Polygon(points)

            # If polygon is not valid, try to make it valid using make_valid from shapely. This will attempt to fix
            # common issues like self-intersections, but it may result in a MultiPolygon or GeometryCollection if the
            # original polygon is too malformed. We will need to handle those cases as well.
            if not poly.is_valid and not ignore_invalid:
                logging.warning(explain_validity(poly))

                if custom_validation:
                    poly = poly.buffer(0)
                    if not poly.is_valid or poly.area == 0:
                        logging.warning(f"Custom validation failed for polygon: {polygon}")
                        continue
                    # Handle case where buffer returns MultiPolygon
                    # (take largest component rather than splitting)
                    if hasattr(poly, "geoms"):
                        logging.warning(f"Custom validation created MultiPolygon, taking the largest.")
                        poly = max(poly.geoms, key=lambda p: p.area)

                    valid_polygons.append(poly)

                else:
                    clean_poly = make_valid(poly)

                    # Check if it's already a single Polygon
                    if isinstance(clean_poly, Polygon):
                        valid_polygons.append(clean_poly)

                    # If it's a collection (MultiPolygon or GeometryCollection), extract Polygons
                    elif hasattr(clean_poly, "geoms"):
                        polygons = [geom for geom in clean_poly.geoms if isinstance(geom, Polygon)]
                        valid_polygons.extend(polygons)

            # If the original polygon is valid, we can keep it as is.
            else:
                valid_polygons.append(poly)

        return valid_polygons

    def _process_single_image(self, image_path):
        """Processes a single image to generate pseudo labels.
        :param image_path: string, path to the image
        :return: dictionary, pseudo label segmentations in coco format along with the width and height of original image
        """
        # Raise an error if the image path does not exist
        if not os.path.exists(image_path):
            logging.error("Image does not exist: %s", image_path)
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Load the image
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Convert the image to a numpy array
        image_np = np.array(image)

        width, height = image.size

        logging.info("Opened image: %s", image_path)
        logging.info("Image size: %dx%d", width, height)

        # apply threshold
        thresh = threshold_otsu(image_np)
        bw = closing(image_np > thresh, footprint_rectangle((5, 5), decomposition='sequence')) \
            if self.threshold_direction == 1 else (
            closing(image_np < thresh, footprint_rectangle((5, 5), decomposition='sequence')))

        # Print and save the thresholded image
        self._print_and_save_output_image(bw, image_path, image_suffix='thresholded', cmap='gray')

        # remove artifacts connected to image border
        cleared = cleanup_mask(bw, buffer_size=0, print_output=False, image=image_np, min_object_size=0,
                               min_hole_size=5)

        # Print and save the cleared image
        self._print_and_save_output_image(cleared, image_path, image_suffix='cleared', cmap='gray')

        # label image regions
        label_image = label(cleared)

        # Print the number of labels found
        num_labels = np.max(label_image)
        logging.info("Number of initial labels: %d", num_labels)

        # Print and save the labeled image
        self._print_and_save_output_image(label_image, image_path, image_suffix='label', cmap='nipy_spectral')

        # Dilate the labeled image to connect nearby regions
        dilated_image = morphology.binary_dilation(cleared, footprint_rectangle((3, 3), decomposition='sequence'))

        # Print and save the dilated image
        self._print_and_save_output_image(dilated_image, image_path, image_suffix='dilated', cmap='gray')

        # Clear again
        cleared = cleanup_mask(dilated_image, buffer_size=0, print_output=False, image=image_np, min_object_size=5,
                               min_hole_size=10)

        # Print and save the cleared image after dilation
        self._print_and_save_output_image(cleared, image_path, image_suffix='cleared_after_dilation', cmap='gray')

        # Label image regions
        label_image = label(cleared)

        # Print and save the labeled image after dilation
        self._print_and_save_output_image(label_image, image_path, image_suffix='label_after_dilation', cmap='nipy_spectral')

        # With the label_image, we can see the boundaries of ROIs. We want to create mask with everything inside the boundaries.
        # To achieve that we will use fill_holes to fill the empty spaces inside the boundaries.
        object_masks = {}
        combined_mask = np.zeros_like(label_image, dtype=bool)

        for region in regionprops(label_image):
            if region.area < 100:  # Skip small regions
                continue
            region_label = region.label
            min_row, min_col, max_row, max_col = region.bbox

            # Create a blank mask for the entire image
            mask = np.zeros_like(label_image, dtype=bool)

            # Create a slice of the label image and find mask for current label
            region_slice = label_image[min_row:max_row, min_col:max_col]
            rows, cols = np.where(region_slice == region_label)

            # Offset coordinates to fit the main image
            rows += min_row
            cols += min_col

            # Set the mask inside the bounding box to True where label matches
            mask[rows, cols] = True

            # Fill holes in the mask
            structure = np.ones((3, 3), dtype=bool)  # 3x3 structure works best for filling smaller holes.
            filled_mask = ndimage.binary_fill_holes(mask, structure=structure).astype(bool)

            object_masks[region_label] = filled_mask

            # Combine the mask with the main mask
            combined_mask |= filled_mask

        # Print and save the combined mask of all objects
        self._print_and_save_output_image(combined_mask, image_path, image_suffix='combined_mask', cmap='gray')

        # Use our custom region growing algorithm to fill the holes and create masks
        regions = find_regions_of_interest(image_np, combined_mask, tolerance=self.tolerance)
        logging.info("Number of regions found: %d", len(regions))
        region_mask = get_region_mask(regions, image=image_np)

        # Print and save the region mask after region growing
        self._print_and_save_output_image(region_mask, image_path, image_suffix='region_mask', cmap='gray')

        # Clean the region mask to remove small objects and holes
        cleaned_mask = cleanup_mask(region_mask, buffer_size=2, print_output=False, image=image_np, min_object_size=150,
                                    min_hole_size=150, try_binary_fill_holes=False)

        # Print and save the cleaned region mask
        self._print_and_save_output_image(cleaned_mask, image_path, image_suffix='cleaned_region_mask', cmap='gray')

        # Create COCO annotations from the cleaned mask
        coco_annotations = self._create_coco_annotation(cleaned_mask)
        logging.info("Generated COCO annotations for image: %s", image_path)
        logging.info("Number of annotations: %d", len(coco_annotations))

        return coco_annotations, width, height

    def _process_batch_images(self, directory_path):
        """Processes a batch of images in a directory to generate pseudo labels.
        :param directory_path: string, path to the directory containing images
        :return: dictionary, mapping of image paths to their corresponding pseudo label segmentations in coco format
        """
        if not os.path.exists(directory_path):
            logging.error("Directory does not exist: %s", directory_path)
            raise FileNotFoundError(f"Directory not found at {directory_path}")

        coco_annotations_dict = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(directory_path, filename)
                logging.info("Processing image: %s", image_path)
                coco_annotations, width, height = self._process_single_image(image_path)
                coco_annotations_dict[image_path] = {
                    'annotations': coco_annotations,
                    'width': width,
                    'height': height
                }

        logging.info("Completed processing batch of images in directory: %s", directory_path)
        return coco_annotations_dict

    def _create_coco_annotation(self, mask, **kwargs):
        """Creates a COCO annotation from a given mask.
        :param mask: numpy array, binary mask of the object
        :return: dictionary, COCO annotation for the object
        """
        # Get the contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotations = []
        annotation_id = 1  # Start annotation ID from 1
        image_id = kwargs.get('image_id', 1)  # Default image ID is 1, can be overridden by kwargs
        category_id = kwargs.get('category_id', 1) # Default category ID is 1, can be overridden by kwargs
        for contour in contours:
            # Flatten the contour points and convert to list
            segmentation = contour.flatten().tolist()

            # Calculate bounding box (x_min, y_min, width, height)
            x, y, w, h = cv2.boundingRect(np.array(segmentation).reshape(-1, 2))
            bbox = [x, y, w, h]
            area = int(w) * int(h)
            if area > 10:
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0
                }
                annotations.append(annotation)
                annotation_id += 1
        return annotations

    def _print_and_save_output_image(self, output, image_path, image_suffix='', cmap='gray'):
        """Helper function to print and save the output image at each step of the pseudo label generation process.
        :param output: numpy, the output to be printed and saved
        :param image_path: string, path to the original image for naming the output file
        """
        # Print the output image if self.print is True
        if self.print:
            plt.figure(figsize=(10, 10))
            plt.imshow(output, cmap=cmap)

        # Save the output image if self.save is True
        if self.save:
            self.saved_image_count += 1
            # If image_suffix is not provided, use the saved_image_count as the suffix for the saved image
            image_suffix = self.saved_image_count if image_suffix == '' else image_suffix

            # Create the save directory if it does not exist
            os.makedirs(self.save_dir, exist_ok=True)
            # Save the output image with a name that includes the original image name and the suffix
            plt.figure(figsize=(10, 10))
            plt.imsave(os.path.join(self.save_dir, f'{os.path.basename(image_path)}_{image_suffix}.png'), output, cmap=cmap)
            plt.close()