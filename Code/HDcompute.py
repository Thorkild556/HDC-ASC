import random
import numpy as np
import math
#import jax.numpy as jnp
#import jax
import torch
# from jax import random as jr
from collections import OrderedDict

def make_rand_vector_torch(size = 10000, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, 2, (length,), dtype=torch.float32) * 2 - 1  # Random vector with values -1 or 1

def make_flip_tensor_batch(vector1, vector2, percentage_matrix, seed=None): #way faster
    if seed is not None:
        torch.manual_seed(seed)

    # Ensure inputs are tensors
    vector1 = torch.as_tensor(vector1, dtype=torch.int8)
    vector2 = torch.as_tensor(vector2, dtype=torch.int8)
    percentage_matrix = torch.as_tensor(percentage_matrix, dtype=torch.float32)
    device = vector1.device

    # Tensor dimensions
    num_rows, num_cols = percentage_matrix.shape
    length = vector1.shape[0]

    # Number of flips for each entry


    # Generate random indices
    flat_random = torch.rand(num_rows * num_cols, length, device=device)  # Generate random values
    threshold = percentage_matrix.flatten().unsqueeze(-1)                 # Use percentage directly
    flip_mask_flat = flat_random < threshold                              # Create mask, where each number in the 3'd dim of the flat random is compared to the corresponding value in the percentage matrix


    # Reshape flip_mask
    flip_mask = flip_mask_flat.view(num_rows, num_cols, length)

    # Apply flipping
    vector1_expanded = vector1.unsqueeze(0).unsqueeze(0)
    vector2_expanded = vector2.unsqueeze(0).unsqueeze(0)
    output_tensor = torch.where(flip_mask, vector2_expanded, vector1_expanded)

    return output_tensor


##actual implementation

def n_gram_encode_tensor_torch(tensor, n, single_window = False, weighted = False):# it works now

    # Clone the input tensor to avoid modifying it
    result = tensor.clone()
    tensor = tensor.clone()
    result = torch.roll(tensor, shifts=(n-1), dims=2) #permute n-1 times
    tensor = torch.roll(tensor, shifts=(n-1), dims=2)


    for i in range(1, n):
        tensor = torch.roll(tensor, shifts=-1, dims=2) # reverse permutation

        result[:, :-i, :] *= tensor[:, i: , :]

    if weighted == True:
        result = compute_weighted_hypervector_n(result, dim=1, emphasis_factor=1)


    if single_window == False:
        #Sum along the sequence length dimension
        sum_matrix = torch.sum(result, dim=1)
        result = sum_matrix

    return result
    # return sum_matrix

import torch

def n_gram_encode_first_n(tensor, n, use_permutation=True):
    # 1) Clone and keep only the first n windows
    #    shape becomes (signal, n, vectorvalues)
    t = tensor.clone()[:, :n, :]
    
    # 2) Create a list of (optionally permuted) slices
    slices = []
    for i in range(n):
        # Each slice has shape (signal, vectorvalues)
        slice_i = t[:, i, :]
        
        if use_permutation:
            # "Positional" permutation in hyperdimensional computing
            # by rolling along the vector dimension
            # dims=1 is correct here because slice_i has shape (signal, vectorvalues).
            slice_i = torch.roll(slice_i, shifts=i, dims=1)
        
        slices.append(slice_i)
    
    # 3) Multiply (bind) all slices together to get one final hypervector per signal
    #    Start with the first slice
    out = slices[0]
    for i in range(1, n):
        out = out * slices[i]  # elementwise multiply

    # 4) Return shape (signal, vectorvalues)
    return out



def normalise_mfccs(mfccs, separate_signals=False): #refactored
    copy_mfccs = mfccs.clone()
    if not separate_signals:
        # Use amin and amax for multidimensional min/max
        min_vals = torch.amin(mfccs, dim=(0, 1), keepdim=True)
        max_vals = torch.amax(mfccs, dim=(0, 1), keepdim=True)
        diff = max_vals - min_vals
        copy_mfccs = (copy_mfccs - min_vals) / diff
    else:
        # Use amin and amax for reduction along specific dimensions
        min_vals = torch.amin(mfccs, dim=1, keepdim=True)
        max_vals = torch.amax(mfccs, dim=1, keepdim=True)
        diff = max_vals - min_vals
        copy_mfccs = (copy_mfccs - min_vals) / diff

    return copy_mfccs

def global_normalise_mfccs(mfccs, separate_signals=False):
    copy_mfccs = mfccs.clone()

    if not separate_signals:
        # Global normalisation across all signals and windows
        min_val = torch.amin(mfccs)  # Single global min
        max_val = torch.amax(mfccs)  # Single global max
        diff = max_val - min_val
        copy_mfccs = (copy_mfccs - min_val) / diff
    else:
        # Normalisation separately for each signal
        min_vals = torch.amin(mfccs, dim=(1, 2), keepdim=True)  # Per signal min
        max_vals = torch.amax(mfccs, dim=(1, 2), keepdim=True)  # Per signal max
        diff = max_vals - min_vals
        copy_mfccs = (copy_mfccs - min_vals) / diff

    return copy_mfccs



#helper method for each batch
def encode_mfcc_batch(mfccs_batch, mfccs_non, num_cols, num_mfccs, length, vector2_expanded, vector1_expanded, device, seed = None, weighing = False):
    #setting seed
    if seed is not None:
        torch.manual_seed(seed)

     # Generate random values for the current batch
    flat_random = torch.rand(
        mfccs_batch.shape[0], num_cols, num_mfccs, length, device=device, dtype=torch.float32
    )

    # Compute threshold and flip mask
    threshold = mfccs_batch.unsqueeze(-1)  # Shape: (batch_size, num_cols, num_mfccs, 1)
    flip_mask = flat_random < threshold  # Shape: (batch_size, num_cols, num_mfccs, length)
    batch_output = torch.where(flip_mask, vector2_expanded, vector1_expanded)

    if weighing == True:
        # Multiply each value in the vector with the corresponding MFCC value
        # mfccs_expanded =  torch.abs(mfccs_non.unsqueeze(-1))  # Shape: (batch_size, num_cols, num_mfccs, 1)
        batch_output = batch_output * torch.abs(mfccs_non.unsqueeze(-1))  # Shape: (batch_size, num_cols, num_mfccs, length)


    # Apply flipping for the current batch
    return batch_output


def create_mfcc_vector_matrix(n_mfccs, vector_length, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate a matrix of random hypervectors with values -1 or 1
    mfcc_vector_matrix = torch.randint(0, 2, (n_mfccs, vector_length), dtype=torch.float32) * 2 - 1
    
    return mfcc_vector_matrix


def bind_and_add_mfcc_vectors(mfcc_encoded_vectors, rand_mfcc_matrix):
    # Ensure the dimensions match
    batch_size, num_windows, num_mfccs, vector_length = mfcc_encoded_vectors.shape
    assert rand_mfcc_matrix.shape == (num_mfccs, vector_length), "Dimension mismatch between rand_mfcc_matrix and mfcc_encoded_vectors"

    # Perform element-wise multiplication
    multiplied = mfcc_encoded_vectors * rand_mfcc_matrix.unsqueeze(0).unsqueeze(0)

    # Sum the results across the MFCC dimension
    summed = multiplied.sum(dim=2)

    return summed


def bind_and_add_time_vectors(encoded_windows, time_vector_matrix):

    multiplied = encoded_windows * time_vector_matrix.unsqueeze(0)

    summed = multiplied.sum(dim=1)

    return summed


def bind_time_vectors(encoded_windows, time_vector_matrix):

    multiplied = encoded_windows * time_vector_matrix.unsqueeze(0)

    return multiplied

def majority_vote_torch(arr):
    device = arr.device  # Ensure arr is on the GPU
    arr = torch.where(arr < 0, torch.tensor(-1, device=device, dtype=arr.dtype), arr)
    arr = torch.where(arr > 0, torch.tensor(1, device=device, dtype=arr.dtype), arr)
    return arr

def log_transform_torch(data, alpha):
    return torch.sign(data) * torch.log1p(alpha * torch.abs(data))

def sqrt_transform_torch(data):
    return torch.sign(data) * torch.sqrt(torch.abs(data))

def group_windows(tensor, group_size):
    # Get the shape of the input tensor
    signals, windows, vector = tensor.shape
    
    # Calculate the number of groups
    num_groups = windows // group_size
    
    # Reshape the tensor to group the windows
    grouped_tensor = tensor[:, :num_groups * group_size, :].reshape(signals, num_groups, group_size, vector)
    
    # Sum the windows within each group
    grouped_tensor = grouped_tensor.sum(dim=2)
    
    return grouped_tensor



def encode_mfccs_batched(mfccs, vector1, vector2, batch_size, rand_mfcc_matrix, separate_signals=False, 
                            seed = None, n_gram = 2, alpha = 1, single_window = False, group_size = False,
                            weighing = False):

    mfccs = log_transform_torch(mfccs, alpha)


    # Normalize MFCCs
    normalised = normalise_mfccs(mfccs=mfccs, separate_signals=separate_signals)


    # Get the device
    device = vector1.device

    # Get tensor sizes
    num_rows, num_cols, num_mfccs = mfccs.shape
    length = vector1.shape[0]

    # Expand vector1 and vector2 to match the dimensions of flip_mask
    vector1_expanded = vector1.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1, length)
    vector2_expanded = vector2.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1, length)

    # Initialize list for storing results
    output_tensors = []

    #set seed
    if seed is not None:
        torch.manual_seed(seed)

    # if bool(n_gram) == False:
    #     time_vector_matrix = torch.randint(0, 2, (num_cols, length), dtype=torch.float32) * 2 - 1
    #     time_vector_matrix = time_vector_matrix.to(device)

    # Process in batches
    for start_idx in range(0, num_rows, batch_size):
        end_idx = min(start_idx + batch_size, num_rows)
        seed += 1
        
        # Extract the batch
        mfccs_batch = normalised[start_idx:end_idx]  # Shape: (batch_size, num_cols, num_mfccs)
        mfccs_log = mfccs[start_idx:end_idx]

        encoded_batch = encode_mfcc_batch(mfccs_batch, mfccs_log, num_cols, num_mfccs, length, vector2_expanded, vector1_expanded, device, seed = seed, weighing = weighing)

        bound_batch = bind_and_add_mfcc_vectors(encoded_batch, rand_mfcc_matrix)

        if bool(group_size) == True:
            bound_batch = group_windows(bound_batch, group_size)



        bound_batch = majority_vote_torch(bound_batch)

            # print(torch.count_nonzero(bound_batch== 1))
            # print(torch.count_nonzero(bound_batch== -1))
            # print(bound_batch.shape)
            
            # break

        finished = n_gram_encode_tensor_torch(bound_batch, n= n_gram, single_window = single_window)
        #
        # finished = n_gram_encode_first_n(bound_batch, n= n_gram, use_permutation=True)

        output_tensors.append(finished)

    output_tensor = torch.cat(output_tensors, dim=0) # Shape: (signals, vector)

    # if weighted == True:
    #     output_tensor = compute_weighted_hypervector(output_tensor, dim = 0)

    return output_tensor



def make_model_from_torch_matrix(torch_matrix, target, single_window = False):
    device = "mps"
    torch_matrix = torch_matrix.to(device)
    unique_labels = torch.unique(target)
    n_target = len(unique_labels)
    if single_window == True:
        model = torch.zeros((n_target, torch_matrix.shape[1], torch_matrix.shape[2]), dtype=torch.float32).to(torch_matrix.device)
    elif single_window == False:
        model = torch.zeros((n_target, torch_matrix.shape[1]), dtype=torch.float32).to(torch_matrix.device)

    for idx, label in enumerate(unique_labels):
        mask = (target == label)
        model[idx] = torch.sum(torch_matrix[mask], dim=0)

    return model, unique_labels


def make_model_from_mfccs(mfccs, target, vector1, vector2, batch_size, device, separate_signals=True, seed = None, n_gram = 5, majority_vote = False, alpha = 1, single_window = False, group_size = False, weighing = False):
    #creating my mfcc_matrix to use for every batch to bind the vector values for each mffc #
    rand_mfcc_matrix = create_mfcc_vector_matrix(n_mfccs = mfccs.shape[2], vector_length = vector1.shape[0], seed=seed).to(device)
    
    encoded_mfccs = encode_mfccs_batched(mfccs, vector1, vector2, batch_size, rand_mfcc_matrix, separate_signals=separate_signals, seed = seed, n_gram = n_gram, alpha = alpha, single_window = single_window, group_size = group_size, weighing = weighing)

    if majority_vote == True:
        encoded_mfccs = majority_vote_torch(encoded_mfccs)

    model, unique_labels = make_model_from_torch_matrix(encoded_mfccs, target, single_window= single_window)

    return model, unique_labels, rand_mfcc_matrix, vector1, vector2, batch_size, separate_signals, seed, n_gram, alpha, group_size



def cosine_similarity_matrix_torch(M, X):

    # First Transpose X to get Xt
    Xt = X.T
    
    # Calculate the dot product Of M and Xt
    dot_product = torch.mm(M, Xt)
    
    # Calculate the norms or the lengths
    M_norm = torch.norm(M, dim=1).unsqueeze(1)
    Xt_norm = torch.norm(Xt, dim=0).unsqueeze(0)
    
    # Calculate the outer product of the norms with pairwise multiplication
    norm_product = M_norm * Xt_norm
    
    # Calculate the cosine similarity with pairwise division
    cosine_similarity = dot_product / norm_product
    
    return cosine_similarity


def predict(model, mfccs, majority_vote = False, weighing = False):
    mod, unique_labels, rand_mfcc_matrix, vector1, vector2, batch_size, separate_signals, seed, n_gram, alpha, group_size = model

    seed += 1

    encoded_mfccs = encode_mfccs_batched(mfccs, vector1, vector2, batch_size, rand_mfcc_matrix, separate_signals=separate_signals, seed = seed, n_gram = n_gram, alpha=alpha, group_size=group_size, weighing = weighing)
    
    if majority_vote == True:
        encoded_mfccs = majority_vote_torch(encoded_mfccs)

    cos_sim_matrix = cosine_similarity_matrix_torch(mod, encoded_mfccs)

    #i have no freakin' clue why this is neccesary but the gpu returns the wrong argmax for some reason
    cos_sim_matrix = cos_sim_matrix.to("cpu") 

    argmax = torch.argmax(cos_sim_matrix, dim = 0)

    predictions = unique_labels[argmax]

    return predictions


def fine_tune_model(model, labels, encoded_mfccs, learning_rate = 0.2, iterations = 1, majority_vote = False):

    mod, unique_labels, rand_mfcc_matrix, vector1, vector2, batch_size, separate_signals, seed, n_gram, alpha, group_size = model

    seed += 1

    #encoded_mfccs = encode_mfccs_batched(mfccs, vector1, vector2, batch_size, rand_mfcc_matrix, separate_signals=separate_signals, seed = seed, n_gram = n_gram, alpha=alpha, group_size=group_size, weighing = weighing)
    
    if majority_vote == True:
        encoded_mfccs = majority_vote_torch(encoded_mfccs)

    for i in range(iterations):
        cos_sim_matrix = cosine_similarity_matrix_torch(mod, encoded_mfccs)

        #i have no freakin' clue why this is neccesary but the gpu returns the wrong argmax for some reason
        cos_sim_matrix = cos_sim_matrix.to("cpu") 

        argmax = torch.argmax(cos_sim_matrix, dim = 0)

        predictions = unique_labels[argmax]

        wrong_idx = torch.nonzero(predictions != labels)

        mis_preds = predictions[wrong_idx]      # rows to update
        mis_true  = labels[wrong_idx] 
        mis_vecs  = encoded_mfccs[wrong_idx]    # vectors to subtract

        # Subtract in a single, vectorized operation:
        mod.index_put_((mis_preds,), -mis_vecs*learning_rate, accumulate=True)

        #Add to the true class row
        mod.index_put_((mis_true,), mis_vecs * learning_rate, accumulate=True)

        

    return mod, unique_labels, rand_mfcc_matrix, vector1, vector2, batch_size, separate_signals, seed, n_gram, alpha, group_size



def predict_vote(model, mfccs, majority_vote=False): #summed cosine similarity
    mod, unique_labels, rand_mfcc_matrix, vector1, vector2, batch_size, separate_signals, seed, n_gram, alpha = model

    seed += 1

    # Encode MFCCs into (signals x windows x vector) tensor
    encoded_mfccs = encode_mfccs_batched(mfccs, vector1, vector2, batch_size, rand_mfcc_matrix, separate_signals=separate_signals, seed=seed, n_gram=n_gram, alpha=alpha, single_window=True).to("mps")

    # Initialize an accumulator for class scores (signals x classes)
    class_scores = torch.zeros(mod.shape[0], encoded_mfccs.shape[0], device=encoded_mfccs.device, dtype=torch.float32)

    # Iterate over windows (second dimension)
    for i in range(encoded_mfccs.shape[1]):  # Loop through each window
        # Extract the corresponding window from the model
        model_windows = mod[:, i, :]  # Shape: (classes x vector)

        # Extract the corresponding window across all signals
        signal_windows = encoded_mfccs[:, i, :]  # Shape: (signals x vector)

        # Compute cosine similarity for all signals against all classes for this window
        cos_sim_matrix = cosine_similarity_matrix_torch(model_windows, signal_windows)  # Shape: (classes x signals)
        # Accumulate scores across windows
        class_scores += cos_sim_matrix  # Accumulate directly (no transpose needed)

    class_scores = class_scores.to("cpu")
    # Print the first 10 columns for manual inspection
    # Get the index of the highest scoring class for each signal
    argmax = torch.argmax(class_scores, dim=0)  # Shape: (signals,)


    # Map indices to unique labels
    predictions = unique_labels[argmax]

    return predictions


from collections import Counter

# def predict_vote(model, mfccs, majority_vote=False): #votes
#     mod, unique_labels, rand_mfcc_matrix, vector1, vector2, batch_size, separate_signals, seed, n_gram, alpha = model

#     seed += 1

#     # Encode MFCCs into (signals x windows x vector) tensor
#     encoded_mfccs = encode_mfccs_batched(mfccs, vector1, vector2, batch_size, rand_mfcc_matrix, separate_signals=separate_signals, seed=seed, n_gram=n_gram, alpha=alpha, single_window=True)

#     # Initialize a list to store predictions for each window
#     window_predictions = []

#     # Iterate over windows (second dimension)
#     for i in range(encoded_mfccs.shape[1]):  # Loop through each window
#         # Extract the corresponding window from the model
#         model_windows = mod[:, i, :]  # Shape: (classes x vector)

#         # Extract the corresponding window across all signals
#         signal_windows = encoded_mfccs[:, i, :]  # Shape: (signals x vector)

#         # Compute cosine similarity for all signals against all classes for this window
#         cos_sim_matrix = cosine_similarity_matrix_torch(model_windows, signal_windows)  # Shape: (classes x signals)

#         cos_sim_matrix = cos_sim_matrix.to("cpu")

#         # Get the index of the highest scoring class for each signal
#         argmax = torch.argmax(cos_sim_matrix, dim=0)  # Shape: (signals,)

#         # Map indices to unique labels and store predictions
#         window_predictions.append(unique_labels[argmax])

#     # Convert list of predictions to a tensor
#     window_predictions = torch.stack(window_predictions, dim=1)  # Shape: (signals, windows)

#     # Determine the majority vote for each signal
#     final_predictions = []
#     for signal_predictions in window_predictions:
#         most_common = Counter(signal_predictions.tolist()).most_common(1)[0][0]
#         final_predictions.append(most_common)

#     return torch.tensor(final_predictions)



def compute_weighted_hypervector(model_tensor, dim = 0, emphasis_factor = 2):
    # Step 1: Compute standard deviation across classes for each dimension
    std_per_dimension = torch.std(model_tensor, dim=0)  # Shape: (vector_dim,)

    std_per_dimension = std_per_dimension ** emphasis_factor

    # Step 2: Normalize std to create weights
    weights = std_per_dimension / torch.sum(std_per_dimension)  # Normalize to sum to 1

    # Step 3: Apply weights to each hypervector
    weighted_hypervectors = model_tensor * weights  # Element-wise multiplication

    return weighted_hypervectors


def compute_weighted_hypervector_n(model_tensor, dim=1, emphasis_factor=1):
    # Step 1: Compute standard deviation across the specified dimension
    std_per_dimension = torch.std(model_tensor, dim=dim)  # Shape: (signal, vector)

    # Step 2: Apply emphasis factor
    std_per_dimension = std_per_dimension ** emphasis_factor

    # Step 3: Normalize std to create weights
    weights = std_per_dimension / torch.sum(std_per_dimension, dim=-1, keepdim=True)  # Shape: (signal, vector)

    # Step 4: Apply weights to the tensor
    # Expand weights to match model_tensor shape
    expanded_weights = weights.unsqueeze(dim)  # Shape: (signal, 1, vector)
    weighted_hypervectors = model_tensor * expanded_weights  # Element-wise multiplication

    return weighted_hypervectors


def encode_mfcc_batch_with_percentage_flip(
    mfccs_batch, num_cols, num_mfccs, vector_length, 
    min_vector, mean_vector, max_vector, device, seed=None
):
    if seed is not None:
        torch.manual_seed(seed)

    # Calculate mean for each MFCC across time windows for each signal
    mean_vals = torch.mean(mfccs_batch, dim=1, keepdim=True)  # Shape: (signals, 1, mfccs)

    # Calculate range (max - mean or mean - min) and avoid division by zero
    range_vals = torch.amax(mfccs_batch, dim=1, keepdim=True) - torch.amin(mfccs_batch, dim=1, keepdim=True) + 1e-6

    # Calculate percentage deviation from the mean
    percentages = (mfccs_batch - mean_vals) / range_vals  # Shape: (signals, windows, mfccs)
    percentages = percentages.clamp(min=-1, max=1)  # Clamp to range [-1, 1]

    # Expand percentage tensor to match hypervector dimensions
    percentages_expanded = (percentages * 2).clamp(min=-1, max=1).unsqueeze(-1)  # Amplify influence


    # Expand vectors to match batch dimensions
    mean_vector_expanded = mean_vector.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, mfccs, vector_length)
    min_vector_expanded = min_vector.unsqueeze(0).unsqueeze(0)
    max_vector_expanded = max_vector.unsqueeze(0).unsqueeze(0)

    # Generate random values for flipping
    random_values = torch.rand(mfccs_batch.shape[0], num_cols, num_mfccs, vector_length, device=device)

    # Calculate masks for flipping towards min and max
    flip_towards_min = percentages_expanded < 0  # True if deviation is below the mean
    flip_towards_max = percentages_expanded > 0  # True if deviation is above the mean

    # Compute absolute percentages for blending
    abs_percentages = percentages_expanded.abs()

    # Generate the final flipped vectors
    flipped_towards_min = torch.where(
        random_values < abs_percentages,
        min_vector_expanded,
        mean_vector_expanded
    )
    flipped_towards_max = torch.where(
        random_values < abs_percentages,
        max_vector_expanded,
        mean_vector_expanded
    )

    # Combine based on masks
    flipped_vectors = torch.where(
        flip_towards_min,
        flipped_towards_min,
        flipped_towards_max
    )

    return flipped_vectors
