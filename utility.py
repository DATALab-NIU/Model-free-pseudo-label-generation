# This file contains different utility functions used by the PseudoLabelAnnotation class.
# Author: Ashiqur Rahman
# URL: https://github.com/ashiqur-rony

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import clear_border
from scipy.optimize import linear_sum_assignment
from skimage.measure import label
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from collections import deque


def _decompose_size(size, kernel_size=3):
    """Determine number of repeated iterations for a `kernel_size` kernel.

    Returns how many repeated morphology operations with an element of size
    `kernel_size` is equivalent to a morphology with a single kernel of size
    `n`.

    """
    if kernel_size % 2 != 1:
        raise ValueError("only odd length kernel_size is supported")
    return 1 + (size - kernel_size) // (kernel_size - 1)


def footprint_rectangle(shape, *, dtype=np.uint8, decomposition=None):
    """Generate a rectangular or hyper-rectangular footprint.

    Generates, depending on the length and dimensions requested with `shape`,
    a square, rectangle, cube, cuboid, or even higher-dimensional versions
    of these shapes.

    Parameters
    ----------
    shape : tuple[int, ...]
        The length of the footprint in each dimension. The length of the
        sequence determines the number of dimensions of the footprint.
    dtype : data-type, optional
        The data type of the footprint.
    decomposition : {None, 'separable', 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        give an identical result to a single, larger footprint, but often with
        better computational performance. See Notes for more details.
        With 'separable', this function uses separable 1D footprints for each
        axis. Whether 'sequence' or 'separable' is computationally faster may
        be architecture-dependent.

    Returns
    -------
    footprint : array or tuple[tuple[ndarray, int], ...]
        A footprint consisting only of ones, i.e. every pixel belongs to the
        neighborhood. When `decomposition` is None, this is just an array.
        Otherwise, this will be a tuple whose length is equal to the number of
        unique structuring elements to apply (see Examples for more detail).

    Examples
    --------
    >>> import skimage as ski
    >>> ski.morphology.footprint_rectangle((3, 5))
    array([[1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1]], dtype=uint8)

    Decomposition will return multiple footprints that combine into a simple
    footprint of the requested shape.

    >>> ski.morphology.footprint_rectangle((9, 9), decomposition="sequence")
    ((array([[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]], dtype=uint8),
      4),)

    `"sequence"` makes sure that the decomposition only returns 1D footprints.

    >>> ski.morphology.footprint_rectangle((3, 5), decomposition="separable")
    ((array([[1],
             [1],
             [1]], dtype=uint8),
      1),
     (array([[1, 1, 1, 1, 1]], dtype=uint8), 1))

    Generate a 5-dimensional hypercube with 3 samples in each dimension

    >>> ski.morphology.footprint_rectangle((3,) * 5).shape
    (3, 3, 3, 3, 3)
    """
    has_even_width = any(width % 2 == 0 for width in shape)
    if decomposition == "sequence" and has_even_width:
        decomposition = "sequence_fallback"

    def partial_footprint(dim, width):
        shape_ = (1,) * dim + (width,) + (1,) * (len(shape) - dim - 1)
        fp = (np.ones(shape_, dtype=dtype), 1)
        return fp

    if decomposition is None:
        footprint = np.ones(shape, dtype=dtype)

    elif decomposition in ("separable", "sequence_fallback"):
        footprint = tuple(
            partial_footprint(dim, width) for dim, width in enumerate(shape)
        )

    elif decomposition == "sequence":
        min_width = min(shape)
        sq_reps = _decompose_size(min_width, 3)
        footprint = [(np.ones((3,) * len(shape), dtype=dtype), sq_reps)]
        for dim, width in enumerate(shape):
            if width > min_width:
                nextra = width - min_width + 1
                component = partial_footprint(dim, nextra)
                footprint.append(component)
        footprint = tuple(footprint)

    else:
        raise ValueError(f"Unrecognized decomposition: {decomposition}")

    return footprint


def find_regions_within_mask(image, mask, tolerance=0):
    """
    Find regions of interest within the mask.
    Parameters:
        :image: numpy array, the input image
        :mask: numpy array, the mask of the bounding boxes
        :tolerance: int, the tolerance for pixel value difference when checking neighbors
    Returns:
        :regions: list of lists, each list contains the coordinates of the pixels in a region
    """
    regions = []
    visited = np.zeros_like(image, dtype=bool)

    # Get the coordinates of the mask
    coords = np.column_stack(np.where(mask))

    # Get the pixel value to compare with
    pixel_value = image[coords[0][0], coords[0][1]]
    found_different_pixel = False
    # Iterate through the mask to find the first pixel that is not part of the mask
    c = 0
    while c in range(len(coords)) and not found_different_pixel:
        row, col = coords[c]
        c += 1
        pixel_value = image[row, col]
        i = 0
        j = 0
        while i in range(image.shape[0] - row) and not found_different_pixel:
            while j in range(image.shape[1] - col) and not found_different_pixel:
                if row + i < image.shape[0] and col + j < image.shape[1]:
                    if not mask[row + i, col + j] and image[row, col] != image[row + i, col + j]:
                        pixel_value = image[row + i, col + j]
                        found_different_pixel = True
                j += 1
            i += 1

    within_mask = False
    queue = []

    # Iterate through each pixel in the image and when we found the start of a mask,
    # we will start a BFS to find all the connected pixels with the same pixel value.
    # This will help us to find the regions of interest inside the bounding boxes.

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if mask[row, col] and not visited[row, col]:
                # We flip the within_mask flag to check if the pixel is within the mask or not
                within_mask = not within_mask

            current_region = []

            # If we are within a mask or the pixel is part of the mask border, add it to the queue.
            if within_mask or mask[row, col]:
                queue = [(row, col)]

            while queue:
                r, c = queue.pop(0)
                if visited[r, c]:
                    continue

                visited[r, c] = True
                current_region.append((r, c))

                # Check neighbors in a 2x2 grid around the pixel
                for dr in [-2, -1, 0, 1, 2]:
                    for dc in [-2, -1, 0, 1, 2]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < image.shape[0] and
                                0 <= nc < image.shape[1] and
                                not visited[nr, nc] and
                                within_mask and
                                abs(int(image[nr, nc]) - pixel_value) <= tolerance):
                            queue.append((nr, nc))

            if current_region:
                regions.append(current_region)

    return regions


def find_regions_of_interest(image, mask, tolerance=0):
    """
    Find regions of interest inside the bounding boxes.
    Parameters:
        :image: numpy array, the input image
        :mask: numpy array, the mask of the bounding boxes
        :tolerance: int, the tolerance for pixel value difference when checking neighbors
    Returns:
        :regions: list of lists, each list contains the coordinates of the pixels in a region
    """
    regions = []
    visited = np.zeros_like(image, dtype=bool)

    # Get the coordinates of the mask
    coords = np.column_stack(np.where(mask))

    # Iterate through each pixel in the image and when we found the start of a mask,
    # we will start a BFS to find all the connected pixels with the same pixel value.
    # This will help us to find the regions of interest inside the bounding boxes.

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if mask[row, col] and not visited[row, col]:
                # To avoid the boundary, we will take the pixel value to the right until current pixel
                # and the pixel value to the right are the same.
                # Once we find that we will use current pixel value.

                pixel_value = image[row, col]
                found_different_pixel = False

                for i in range(image.shape[0] - row):
                    for j in range(image.shape[1] - col):
                        if row + i < image.shape[0] and col + j < image.shape[1]:
                            if not mask[row + i, col + j] and image[row, col] != image[row + i, col + j]:
                                pixel_value = image[row + i, col + j]
                                found_different_pixel = True
                                break
                        if found_different_pixel:
                            break
                    if not found_different_pixel:
                        break

                current_region = []
                queue = [(row, col)]

                while queue:
                    r, c = queue.pop(0)
                    if visited[r, c]:
                        continue

                    visited[r, c] = True
                    current_region.append((r, c))

                    # Check neighbors in a 2x2 grid around the pixel
                    for dr in [-2, -1, 0, 1, 2]:
                        for dc in [-2, -1, 0, 1, 2]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < image.shape[0] and
                                    0 <= nc < image.shape[1] and
                                    not visited[nr, nc] and
                                    abs(int(image[nr, nc]) - int(pixel_value)) <= tolerance):
                                queue.append((nr, nc))

                if current_region:
                    regions.append(current_region)

    return regions


def find_regions_within_boundaries(image, mask):
    """
    Finds all contiguous regions of pixels that are located *within* the
    boundaries defined by the `mask`. This version performs a geometric
    fill. The values from the `image` are not used for region segmentation
    itself, only the `mask` defines what is a boundary and what is interior.

    Improvements over the original approach:
    1.  Seed Pixel Selection: Identifies seed pixels for BFS that are
        definitively *inside* the boundary (i.e., `not mask[pixel]`)
        and adjacent to a boundary pixel.
    2.  BFS Logic: Performs a Breadth-First Search (BFS) that expands
        as long as pixels are within image bounds, *not* part of the `mask`,
        and not yet visited by any fill operation.
    3.  Region Content: Ensures that only non-mask (interior) pixels are
        included in the found regions.
    4.  Efficiency: Uses `collections.deque` for efficient queue operations in BFS.

    Parameters:
        image: numpy array, the input image. Not directly used for segmentation logic
               in this version, but its shape is implicitly assumed to match the mask's.
               Provided for API compatibility with the original function.
        mask: numpy array (boolean or 0/1), where True/1 indicates a boundary pixel.
              Shape should be (rows, cols).
    Returns:
        regions: A list of lists. Each inner list contains (row, col) tuples
                 for pixels belonging to a distinct enclosed region.
    """

    if mask.dtype != bool:
        mask = mask.astype(bool)  # Ensure mask is boolean

    # Get dimensions from the mask
    rows, cols = mask.shape

    # visited_fill tracks non-mask pixels that have already been assigned to a region.
    visited_fill = np.zeros_like(mask, dtype=bool)
    regions = []

    # Iterate over each pixel to find boundary pixels.
    # When a boundary pixel is found, we look for an adjacent non-boundary,
    # unvisited pixel to start a new region fill.
    for r_boundary in range(rows):
        for c_boundary in range(cols):
            # Check if the current pixel is part of a boundary
            if mask[r_boundary, c_boundary]:
                # Explore its 4-connected neighbors to find a seed pixel.
                # A seed pixel must be:
                # 1. Inside the image bounds.
                # 2. NOT a boundary pixel itself (i.e., `not mask[seed_r, seed_c]`).
                # 3. Not already part of a previously filled region.
                for dr_seed, dc_seed in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connectivity

                    nr_potential_seed, nc_potential_seed = r_boundary + dr_seed, c_boundary + dc_seed

                    # Check bounds for the potential seed
                    if 0 <= nr_potential_seed < rows and 0 <= nc_potential_seed < cols:
                        # Check if it's an interior pixel and not yet visited
                        if not mask[nr_potential_seed, nc_potential_seed] and \
                                not visited_fill[nr_potential_seed, nc_potential_seed]:

                            # Found a valid seed pixel. Start a BFS to find the entire region.
                            current_region_coords = []
                            q_fill = deque([(nr_potential_seed, nc_potential_seed)])

                            # Mark the seed pixel as visited immediately to prevent re-processing.
                            visited_fill[nr_potential_seed, nc_potential_seed] = True

                            while q_fill:
                                r_curr, c_curr = q_fill.popleft()
                                current_region_coords.append((r_curr, c_curr))

                                # Explore 4-connected neighbors for the fill.
                                for dr_fill, dc_fill in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    next_r, next_c = r_curr + dr_fill, c_curr + dc_fill

                                    # Check bounds for the neighbor
                                    if 0 <= next_r < rows and 0 <= next_c < cols:
                                        # If the neighbor is within bounds, not yet visited,
                                        # and is an interior pixel (not a boundary),
                                        # add it to the queue and mark as visited.
                                        if not visited_fill[next_r, next_c] and \
                                                not mask[next_r, next_c]:
                                            visited_fill[next_r, next_c] = True
                                            q_fill.append((next_r, next_c))

                            # If the region has pixels, add it to the list of regions.
                            if current_region_coords:
                                regions.append(current_region_coords)
    return regions


def find_intensity_regions_within_boundaries(image, mask):
    """
    Finds contiguous regions of similar pixel intensity that are located
    *within* the boundaries defined by the `mask`.

    This version:
    1.  Identifies seed pixels that are *inside* the boundary (not mask)
        and not yet part of a found region.
    2.  For each seed, it records its pixel intensity from the `image`.
    3.  Performs a Breadth-First Search (BFS) that expands as long as
        neighboring pixels are:
        a. Within image bounds.
        b. Not part of the `mask`.
        c. Not yet visited.
        d. Have the *same intensity* as the seed pixel.
    4.  Ensures only non-mask pixels are included in the regions.

    Parameters:
        image: numpy array (2D, grayscale). Pixel intensities from this image
               are used to define region homogeneity.
        mask: numpy array (boolean or 0/1), where True/1 indicates a boundary pixel.
              Shape should be (rows, cols) and match `image.shape`.
    Returns:
        regions: A list of lists. Each inner list contains (row, col) tuples
                 for pixels belonging to a distinct intensity-based region
                 within the boundaries.
    """

    if image.ndim != 2:
        raise ValueError("Image must be 2D (grayscale).")
    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same dimensions.")

    if mask.dtype != bool:
        mask = mask.astype(bool)

    rows, cols = image.shape

    # visited_pixels tracks pixels already assigned to a region.
    visited_pixels = np.zeros_like(mask, dtype=bool)
    found_regions = []

    for r_boundary in range(rows):
        for c_boundary in range(cols):
            if mask[r_boundary, c_boundary]:  # It's a boundary pixel
                # Check its 4-connected neighbors to find a potential seed
                for dr_seed, dc_seed in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

                    nr_potential_seed, nc_potential_seed = r_boundary + dr_seed, c_boundary + dc_seed

                    if 0 <= nr_potential_seed < rows and 0 <= nc_potential_seed < cols:
                        # Seed must be: not a mask pixel AND not yet visited
                        if not mask[nr_potential_seed, nc_potential_seed] and \
                                not visited_pixels[nr_potential_seed, nc_potential_seed]:

                            current_seed_r, current_seed_c = nr_potential_seed, nc_potential_seed
                            seed_intensity = image[current_seed_r, current_seed_c]

                            current_region_coords = []
                            q_bfs = deque([(current_seed_r, current_seed_c)])

                            # Mark the exact seed pixel as visited
                            visited_pixels[current_seed_r, current_seed_c] = True

                            while q_bfs:
                                r_curr, c_curr = q_bfs.popleft()
                                current_region_coords.append((r_curr, c_curr))

                                for dr_bfs, dc_bfs in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    next_r, next_c = r_curr + dr_bfs, c_curr + dc_bfs

                                    if 0 <= next_r < rows and 0 <= next_c < cols:
                                        if not visited_pixels[next_r, next_c] and \
                                                not mask[next_r, next_c] and \
                                                image[next_r, next_c] == seed_intensity:  # Key check

                                            visited_pixels[next_r, next_c] = True
                                            q_bfs.append((next_r, next_c))

                            if current_region_coords:
                                found_regions.append(current_region_coords)
    return found_regions


def find_intensity_range_within_boundaries(image, mask, intensity_tolerance=5):
    """
    Finds contiguous regions of similar pixel intensity that are located
    *within* the boundaries defined by the `mask`, allowing for a specified
    intensity tolerance. Seed finding is optimized to start from pixels
    adjacent to boundaries.

    This version:
    1.  Identifies seed pixels by checking neighbors of `mask` (boundary) pixels.
        A seed must be *inside* the boundary (not mask) and not yet visited.
    2.  For each seed, it records its pixel intensity from the `image`.
    3.  Performs a Breadth-First Search (BFS) that expands as long as
        neighboring pixels are:
        a. Within image bounds.
        b. Not part of the `mask`.
        c. Not yet visited.
        d. Have an intensity within `intensity_tolerance` of the seed pixel's intensity.
    4.  Ensures only non-mask pixels are included in the regions.

    Parameters:
        image: numpy array (2D, grayscale). Pixel intensities from this image
               are used to define region homogeneity. Assumed to be of a
               numeric type (e.g., uint8, int16, float).
        mask: numpy array (boolean or 0/1), where True/1 indicates a boundary pixel.
              Shape should be (rows, cols) and match `image.shape`.
        intensity_tolerance: int, the maximum allowed absolute difference
                             between a pixel's intensity and the seed pixel's
                             intensity for it to be included in the region.
                             Default is 5.
    Returns:
        regions: A list of lists. Each inner list contains (row, col) tuples
                 for pixels belonging to a distinct intensity-based region
                 within the boundaries.
    """

    if image.ndim != 2:
        raise ValueError("Image must be 2D (grayscale).")
    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same dimensions.")

    if mask.dtype != bool:
        mask = mask.astype(bool)

    rows, cols = image.shape

    # visited_pixels tracks pixels already assigned to a region.
    visited_pixels = np.zeros_like(mask, dtype=bool)
    found_regions = []

    # Iterate through all pixels to find boundary pixels
    for r_boundary in range(rows):
        for c_boundary in range(cols):
            # If it's a boundary pixel, check its neighbors for potential seeds
            if mask[r_boundary, c_boundary]:
                # Explore 4-connected neighbors of the boundary pixel
                for dr_seed, dc_seed in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

                    current_seed_r, current_seed_c = r_boundary + dr_seed, c_boundary + dc_seed

                    # Check if the potential seed is within bounds
                    if 0 <= current_seed_r < rows and 0 <= current_seed_c < cols:
                        # Seed must be: not a mask pixel AND not yet visited
                        if not mask[current_seed_r, current_seed_c] and \
                                not visited_pixels[current_seed_r, current_seed_c]:

                            # Found a valid seed pixel.
                            # It's crucial to handle image dtype correctly for subtraction.
                            seed_intensity = int(image[current_seed_r, current_seed_c])

                            current_region_coords = []
                            q_bfs = deque([(current_seed_r, current_seed_c)])

                            # Mark the exact seed pixel as visited
                            visited_pixels[current_seed_r, current_seed_c] = True

                            while q_bfs:
                                r_curr, c_curr = q_bfs.popleft()
                                current_region_coords.append((r_curr, c_curr))

                                # Explore 4-connected neighbors for BFS expansion
                                for dr_bfs, dc_bfs in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    next_r, next_c = r_curr + dr_bfs, c_curr + dc_bfs

                                    if 0 <= next_r < rows and 0 <= next_c < cols:
                                        # Check if neighbor is valid for region
                                        if not visited_pixels[next_r, next_c] and \
                                                not mask[next_r, next_c]:

                                            current_pixel_intensity = int(image[next_r, next_c])
                                            # Check intensity tolerance
                                            if abs(current_pixel_intensity - seed_intensity) <= intensity_tolerance:
                                                visited_pixels[next_r, next_c] = True
                                                q_bfs.append((next_r, next_c))

                            if current_region_coords:
                                found_regions.append(current_region_coords)
    return found_regions


def fill_gapped_objects(binary_image_array: np.ndarray,
                        closing_iterations: int = 1,
                        connectivity: int = 2) -> np.ndarray:
    """
    Sets all pixels within white boundaries to True in a binary NumPy array,
    attempting to close gaps in boundaries first using morphological closing,
    and then performing hole filling.

    Args:
        binary_image_array: A 2D NumPy array where True represents white pixels
                            (including boundaries and object interiors if any)
                            and False represents black pixels. The function will
                            convert it to a boolean array if it's not already.
        closing_iterations: Number of iterations for the binary closing operation.
                            A larger number of iterations can close larger gaps
                            but may also distort the shapes more or merge very
                            close objects. Start with 1 or 2 for small gaps.
        connectivity: Defines the neighborhood for morphological operations (and hole filling).
                      For a 2D image (which this function expects):
                      - 1 creates a 4-connected structuring element (diamond shape:
                        connects to pixels directly up, down, left, right).
                      - 2 creates an 8-connected structuring element (square shape:
                        connects to all 8 neighboring pixels).
                      Using connectivity=2 (8-connectivity) is generally more robust
                      for closing various gap orientations and ensuring complete hole filling.

    Returns:
        A 2D NumPy array where all objects defined by the (now hopefully closed)
        white boundaries have their interiors set to True.
    """
    if not isinstance(binary_image_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if binary_image_array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    if connectivity not in [1, 2]:  # Assuming 2D input based on ndim check
        raise ValueError("Connectivity must be 1 (for 4-way) or 2 (for 8-way) for 2D images.")

    # Ensure the input array is boolean, as required by morphological functions.
    # This handles cases where the input might be 0s and 1s instead of True/False.
    binary_image_boolean = binary_image_array.astype(bool)

    # 1. Attempt to close gaps in the boundaries using morphological closing.
    #    A structuring element defines the neighborhood for the operation.
    #    - rank = binary_image_boolean.ndim (which is 2 for a 2D image)
    #    - connectivity (1 or 2 as per function arg) determines its shape.
    structuring_element = generate_binary_structure(rank=binary_image_boolean.ndim,
                                                    connectivity=connectivity)

    # Perform binary closing. More iterations close larger gaps.
    # border_value=False ensures that operations near the border treat pixels
    # outside the image as False, preventing artificial connections to the border.
    closed_image = binary_closing(binary_image_boolean,
                                  structure=structuring_element,
                                  iterations=closing_iterations,
                                  border_value=False)

    # 2. Fill the holes in the (hopefully) now-closed objects.
    #    Using the same structuring element (or at least same connectivity) for
    #    hole filling ensures consistency if any fine details of the boundary
    #    matter for defining the hole.
    filled_image_array = binary_fill_holes(closed_image,
                                           structure=structuring_element)

    return filled_image_array


def get_region_mask(regions, image=None):
    if image is None:
        raise ValueError("Image must be provided to visualize regions.")
    region_mask = np.zeros_like(image, dtype=bool)
    flatten_regions = [pixel for region in regions for pixel in region]
    # For each pixel of flatten_regions, set the corresponding pixel in the region_mask to True
    for r, c in flatten_regions:
        region_mask[r, c] = True
    return region_mask


def cleanup_mask(mask, buffer_size=0, print_output=False, image=None, min_object_size=0, min_hole_size=0, try_binary_fill_holes=False, overlay_opacity=1.0):
    """
    This function takes the mask and cleanup by binary filling holes and then cleaning boundaries.
    Parameters:
        mask: numpy array, the input mask
        buffer_size: int, the size of the buffer to clear the border
        print_output: bool, whether to print the output or not
        image: numpy array, the input image to overlay the mask on (optional)
        min_object_size: int, minimum size of objects to keep
        min_hole_size: int, minimum size of holes to fill
        try_binary_fill_holes: bool, whether to try binary filling holes or not
        overlay_opacity: float, opacity of overlay mask
    Returns:
        cleared: numpy array, the cleaned mask after filling holes and clearing borders
    """
    # Binary fill holes in the mask to fill any holes inside the regions
    if try_binary_fill_holes:
        filled_mask = ndimage.binary_fill_holes(mask).astype(bool)
    else:
        filled_mask = mask
    cleared = clear_border(filled_mask, buffer_size=buffer_size)
    cleared = remove_small_objects(cleared, min_object_size)
    cleared = remove_small_holes(cleared, min_hole_size)

    if print_output:
        print(f"Filled mask shape: {filled_mask.shape}, Data type: {filled_mask.dtype}")
        cmap = mcolors.ListedColormap([(0, 0, 0, 0), (1, 1, 1, 1)])  # Transparent black, opaque white
        fig, ax = plt.subplots(figsize=(10, 10))
        if image is not None:
            ax.imshow(image, cmap='gray')
        ax.imshow(cleared, cmap=cmap, alpha=overlay_opacity)  # Overlay the filled mask on the image
    return cleared


def print_regions(regions, figsize=(10, 10), image=None):
    if image is None:
        raise ValueError("Image must be provided to visualize regions.")
    # Add a white mask on the regions of interest
    region_mask = get_region_mask(regions, image)
    print(f"Number of unique labels found: {len(regions)}")
    colors = [(0, 0, 0, 0), (1, 1, 1, 1)]  # Transparent black, opaque white
    cmap = mcolors.ListedColormap(colors)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')

    ax.imshow(region_mask, cmap=cmap, alpha=1.0)  # For the internal regions
    # ax.imshow(combined_mask, cmap=cmap, alpha=1.0) # For the boundary
    return region_mask


def print_region_boundaries(region_properties, image=None, region_threshold=0, show_mask=False, mask_image=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    if image is None:
        raise ValueError("Image must be provided to visualize regions.")
    ax.imshow(image, cmap='gray')

    if show_mask and mask_image is not None:
        # If a mask image is provided, display it
        cmap = mcolors.ListedColormap([(0, 0, 0, 0), (1, 1, 1, 1)])
        ax.imshow(mask_image, cmap=cmap, alpha=0.5)  # Overlay the mask on the image

    for region in region_properties:
        # take regions with large enough areas
        # can adjust
        if region.area >= region_threshold:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor='red',
                linewidth=2,
            )
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def visualize_annotations(image, annotations):
    """
    Visualizes the image with bounding boxes and segmentation polygons from annotations.
    Parameters:
        image: numpy array, the input image to visualize
        annotations: list of dicts, each containing 'bbox' and 'segmentation'
                     where 'bbox' is a list [x, y, width, height]
                     and 'segmentation' is a list of polygons.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    for annotation in annotations:
        bbox = annotation['bbox']
        rect = mpatches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            fill=False,
            edgecolor='red',
            linewidth=2,
        )
        # Create polygon from segmentation
        segmentation = annotation['segmentation']
        if segmentation:
            for seg in segmentation:
                poly = mpatches.Polygon(np.array(seg).reshape(-1, 2), closed=True, fill=False, edgecolor='blue',
                                        linewidth=1)
                ax.add_patch(poly)

        ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def calculate_precision_recall_at_iou(pred_mask, gt_mask, iou_threshold):
    """
    Calculates Precision and Recall for a specific IoU threshold
    by matching connected components (instances).
    """
    # 1. Label connected components to identify individual instances
    # background=0 means the 0-value pixels are background
    pred_labeled, num_preds = label(pred_mask, background=0, return_num=True)
    gt_labeled, num_gts = label(gt_mask, background=0, return_num=True)

    # Edge Case: Both masks are empty
    if num_preds == 0 and num_gts == 0:
        return 1.0, 1.0 # Perfect match (nothing predicted, nothing there)

    # Edge Case: Prediction empty but GT not, or vice versa
    if num_preds == 0 or num_gts == 0:
        return 0.0, 0.0

    # 2. Compute IoU Matrix between every Pred instance and GT instance
    iou_matrix = np.zeros((num_preds, num_gts))

    for i in range(1, num_preds + 1):
        p_mask = (pred_labeled == i)
        for j in range(1, num_gts + 1):
            g_mask = (gt_labeled == j)

            intersection = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()

            if union > 0:
                iou_matrix[i-1, j-1] = intersection / union

    # 3. Match Predictions to Ground Truth (Hungarian Algorithm)
    # We negate IoU because linear_sum_assignment minimizes cost
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # 4. Count True Positives (matches that exceed the threshold)
    true_positives = 0
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            true_positives += 1

    # 5. Calculate Metrics
    # Precision = TP / (TP + FP) -> TP / Total Predictions
    precision = true_positives / num_preds

    # Recall = TP / (TP + FN) -> TP / Total Ground Truths
    recall = true_positives / num_gts

    return precision, recall

def compute_equivalent_diameter_um_from_coords(polygon_coords, pixel_size_um=0.98):
    """
    polygon_coords: flat list [x0, y0, x1, y1, ...] in normalized pixels
    pixel_size_um: physical size of one pixel in micrometers (0.98 μm/pixel for this dataset reported by Imseeh et al.)
    """
    # Reshape flat list into (x, y) pairs
    points = [(polygon_coords[i], polygon_coords[i+1])
              for i in range(0, len(polygon_coords), 2)]

    if len(points) < 3:
        return None

    # Compute area in pixels² using shapely
    poly = Polygon(points)
    if not poly.is_valid or poly.area == 0:
        return None

    area_px = poly.area

    # Convert area to physical units (µm²)
    area_um2 = area_px * (pixel_size_um ** 2)

    # Equivalent circular diameter: d = 2 * sqrt(area / pi)
    diameter_um = 2 * np.sqrt(area_um2 / np.pi)

    return diameter_um

def compute_equivalent_diameter_um(poly, pixel_size_um=0.98):
    """
    poly: Polygon object from shapely
    pixel_size_um: physical size of one pixel in micrometers (0.98 μm/pixel for this dataset reported by Imseeh et al.)
    """
    area_px = poly.area

    # Convert area to physical units (µm²)
    area_um2 = area_px * (pixel_size_um ** 2)

    # Equivalent circular diameter: d = 2 * sqrt(area / pi)
    diameter_um = 2 * np.sqrt(area_um2 / np.pi)

    return diameter_um