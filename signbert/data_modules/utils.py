import numpy as np

def mask_transform_identity(seq, R, max_disturbance, no_mask_joint, K, m):
    """
    Apply different types of masking transformations to a sequence of frames.

    This function randomly applies one of several masking operations to a given
    sequence of frames. The types of masking include joint masking, frame masking,
    clip masking, and identity (no change).  

    Parameters:
    seq (numpy.ndarray): A sequence of frames to be masked.

    Returns:
    tuple:
        - numpy.ndarray: The transformed sequence with applied masking.
        - numpy.ndarray: Indices of frames that have been masked.
    """
    # Make a copy of the input sequence to avoid modifying the original
    toret = seq.copy()
    # Calculate the number of frames without masking
    n_frames = (toret != 0.0).all((1,2)).sum()
    # Calculate the total number of frames to mask based on a predefined ratio R
    n_frames_to_mask = int(np.ceil(R * n_frames))
    # Randomly select frame indices to mask
    frames_to_mask = np.random.choice(n_frames, size=n_frames_to_mask, replace=False)
    clipped_masked_frames = []
    for f in frames_to_mask:
        # Grab frame to be masked
        curr_frame = toret[f]
        # Randomly select the type of masking operation
        op_idx = np.random.choice(4) # 0: joint, 1: frame, 2: clip, 3: identity
        
        if op_idx == 0:
            # Apply joint masking
            curr_frame = mask_joint(curr_frame, max_disturbance, no_mask_joint, m)
            toret[f] = curr_frame
        elif op_idx == 1:
            # Apply frame masking
            curr_frame[:] = 0.
            toret[f] = curr_frame
        elif op_idx == 2:
            # Apply clip masking
            curr_frame, masked_frames_idx = mask_clip(f, toret, n_frames, K)
            clipped_masked_frames.extend(masked_frames_idx)
        else:
            # Identity operation (no change)
            pass
    # Compile a list of all masked frames for use in loss calculation
    masked_frames_idx = np.unique(np.concatenate((frames_to_mask, clipped_masked_frames)))
    
    return toret, masked_frames_idx

def mask_transform(seq, R, max_disturbance, no_mask_joint, K, m):
    """
    Apply different types of masking transformations to a sequence of frames.

    This function randomly applies one of several masking operations to a given
    sequence of frames. The types of masking include joint masking, frame masking,
    and clip masking. 

    Parameters:
    seq (numpy.ndarray): A sequence of frames to be masked.

    Returns:
    tuple:
        - numpy.ndarray: The transformed sequence with applied masking.
        - numpy.ndarray: Indices of frames that have been masked.
    """
    # Make a copy of the input sequence to avoid modifying the original
    toret = seq.copy()
    # Calculate the number of frames without masking
    n_frames = (toret != 0.0).all((1,2)).sum()
    # Calculate the total number of frames to mask based on a predefined ratio R
    n_frames_to_mask = int(np.ceil(R * n_frames))
    # Randomly select frame indices to mask
    frames_to_mask = np.random.choice(n_frames, size=n_frames_to_mask, replace=False)
    clipped_masked_frames = []
    for f in frames_to_mask:
        # Grab frame to be masked
        curr_frame = toret[f]
        # Randomly select the type of masking operation
        op_idx = np.random.choice(3) # 0: joint, 1: frame, 2: clip
        if op_idx == 0:
            # Apply joint masking
            curr_frame = mask_joint(curr_frame, max_disturbance, no_mask_joint, m)
            toret[f] = curr_frame
        elif op_idx == 1:
            # Apply frame masking
            curr_frame[:] = 0.
            toret[f] = curr_frame
        else:
            # Apply clip masking
            curr_frame, masked_frames_idx = mask_clip(f, toret, n_frames, K)
            clipped_masked_frames.extend(masked_frames_idx)
    # Compile a list of all masked frames for use in loss calculation
    masked_frames_idx = np.unique(np.concatenate((frames_to_mask, clipped_masked_frames)))
    
    return toret, masked_frames_idx

def mask_clip(frame_idx, seq, n_frames, K):
    """
    Apply clip masking to a sequence of frames.

    Clip masking involves setting a contiguous subset of frames within the sequence to zero.
    This function randomly determines the length of the clip to mask, centered around a given
    frame index, and then applies the masking. 

    Parameters:
    frame_idx (int): Index of the frame around which the clip is centered.
    seq (numpy.ndarray): The sequence of frames to which the masking will be applied.
    n_frames (int): The total number of frames in the sequence.

    Returns:
    tuple:
        - numpy.ndarray: The sequence with the clip masking applied.
        - list: Indices of the frames that have been masked.
    """
    # Randomly decide the number of frames to mask, with a maximum of K frames
    n_frames_to_mask = np.random.randint(2, K+1)
    n_frames_to_mask_half = n_frames_to_mask // 2
    # Calculate the start and end indices for the clip to be masked
    start_idx = frame_idx - n_frames_to_mask_half
    end_idx = frame_idx + (n_frames_to_mask - n_frames_to_mask_half)
    # Adjust the start and end indices if they go beyond the sequence boundaries
    if start_idx < 0:
        diff = abs(start_idx)
        start_idx = 0
        end_idx += diff
    if end_idx > n_frames:
        diff = end_idx - n_frames
        end_idx = n_frames
        start_idx -= diff
    # Generate a list of indices for the frames to be masked
    masked_frames_idx = list(range(start_idx, end_idx))
    # Apply masking by setting the selected frames to zero
    seq[masked_frames_idx] = 0.0

    return seq, masked_frames_idx

def mask_joint(frame, max_disturbance, no_mask_joint, m):
    """
    Apply masking to specific joints in a frame.

    This function selects a random number of joints in the given frame and applies
    either zero-masking or spatial disturbance to these joints. Zero-masking sets
    the joint coordinates to zero, while spatial disturbance adds a random offset to
    the coordinates.

    Parameters:
    frame (numpy.ndarray): The frame (array of joint coordinates) to be masked.

    Returns:
    numpy.ndarray: The frame with masking applied to specific joints.
    """
    # Define a function for spatial disturbance
    def spatial_disturbance(xy):
        # Add a random disturbance within the range [-max_disturbance, max_disturbance]
        return xy + [np.random.uniform(-max_disturbance, max_disturbance), np.random.uniform(-max_disturbance, max_disturbance)]
    
    # Randomly decide the number of joints to mask, with a maximum of 'm'
    m = np.random.randint(1, m+1)
    # Randomly select joint indices to mask
    joint_idxs_to_mask = np.random.choice(21, size=m, replace=False)
    # Randomly decide the operation to be applied: zero-masking or spatial disturbance
    op_idx = np.random.binomial(1, p=0.5, size=m).reshape(-1, 1)
    # Apply the chosen masking operation to the selected joints
    frame[joint_idxs_to_mask] = np.where(
        op_idx, 
        spatial_disturbance(frame[joint_idxs_to_mask]), 
        spatial_disturbance(frame[joint_idxs_to_mask]) if no_mask_joint else 0.0
    )

    return frame