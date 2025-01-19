import random
import numpy as np
import math
#import jax.numpy as jnp
#import jax
import torch
# from jax import random as jr
from collections import OrderedDict



def make_rand_vector(size=10000, seed=None): # np is faster
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=size, dtype=np.int8) * 2 - 1)

def make_rand_vector_torch(size = 10000, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, 2, (length,), dtype=torch.float32) * 2 - 1  # Random vector with values -1 or 1


def rotate_vector(vec, num_positions):
    return np.concatenate((vec[num_positions:], vec[:num_positions]))


def NGramVector(list_of_vecs = False, degree = False):
    if degree == False:
        print("no degree specified")
        return
    fin_NgramVector = []
    fac = 0 # for when there is not enough data left to make the ngram with n degrees.
    for v in range(len(list_of_vecs)):
        rotated_list = []
        counter = 1
        
        if len(list_of_vecs)-degree < v: #if a full n-gram cannot be made then make a 1 smaller n-gram
                fac += 1
        for i in range(v, v+degree-fac, 1): #increment by one but start at v.
            # print(v)
            rotated_list.append(rotate_vector(list_of_vecs[i], num_positions= degree-counter-fac))
            counter += 1
        ngram_vec = 1
        # print(rotated_list)
        for z in rotated_list:
            ngram_vec = ngram_vec * z
        ngram_vec = jnp.array(ngram_vec)
        fin_NgramVector.append(ngram_vec)
        # print(fin_NgramVector)
    fin_NgramVector = Addvectors(fin_NgramVector)
    return fin_NgramVector


def Addvectors(list_of_vecs = False):
    final_vec = 0
    for i in list_of_vecs:
        final_vec = jnp.add(i, final_vec)

    return final_vec

def Bind(list_of_vecs = False):
    final_vec = 1 #because first vector will become itself when multiplying
    for i in list_of_vecs:
        final_vec = jnp.multiply(i, final_vec)

    return final_vec

def Majority_vote(arr):
    random_values = random_values = random_values = make_rand_vector(size=arr.shape, seed=None)

    # Apply conditions
    arr = arr.at[arr <= -1].set(-1)
    arr = arr.at[arr >= 1].set(1)

    return arr


def flip(vector1, vector2, percentage, seed=None):
    rng = np.random.default_rng(seed)
    num_flips = int(percentage * len(vector1))
    selected_indices = rng.permutation(len(vector1))[:num_flips]
    fin_vector = jnp.copy(vector1)
    fin_vector = fin_vector.at[selected_indices].set(vector2[selected_indices]) # Flip the selected elements

    return fin_vector

def hamming(vector1, vector2): # goes from 0 to 1
    vec_list = [vector1, vector2]
    xor = Bind(vec_list) #binding = XOR-operation when using -1 and 1 in vector.
    hamming_dist = jnp.count_nonzero(xor == 1)/len(vector1)

    return hamming_dist

def cos_sim(vector1, vector2): #cos_sim goes from -1 to 1
    cosine_sim = sum(vector1*vector2)/(math.sqrt(sum(vector1*vector1))*math.sqrt(sum(vector2*vector2)))
    return cosine_sim


#with matrices instead

def make_percentage_matrix(data, seperate_signals = False):
    data1 = jnp.asarray(data, dtype=np.float32)
    if seperate_signals == False:
        max = jnp.max(data)
        min = jnp.min(data)
        diff = max-min
        data1 = (data1 - min)/diff
        return np.asarray((data - min)/diff, dtype=np.float32) #for each value in the data


    if seperate_signals == True:
        for i, row in enumerate(data):
            max = jnp.max(row)
            min = jnp.min(row)
            diff = max-min
            data1 = data1.at[i].set((row - min)/diff) #for each value in the row
        return np.asarray(data1, dtype=np.float32)



def make_flip_tensor(vector1, vector2, percentage_matrix, seed=None): #needs to be optimized
    jax.config.update("jax_enable_x64", True)

    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility

    num_rows, num_cols = percentage_matrix.shape  # Shape of the percentage matrix
    length = vector1.shape[0]  # Length of the input vectors

    # Initialize output tensor with copies of vector1
    output_tensor = np.tile(vector1, (num_rows, num_cols, 1))

    # Calculate the number of indices to flip for each entry
    num_flips = (percentage_matrix * length).astype(int)

    # Prepare a 3D array to hold the selected indices for vector2
    flip_indices = np.zeros((num_rows, num_cols, length), dtype=bool)

    # Create a mask for the positions that need to be flipped
    for i in range(num_rows):
        for j in range(num_cols):
            current_flips = num_flips[i, j]
            if current_flips > 0:
                # Randomly choose indices for flipping without replacement
                chosen_indices = np.random.choice(length, size=current_flips, replace=False) #faster than jax
                flip_indices[i, j, chosen_indices] = True

    # Create a tensor of vector2 to apply the flips directly
    vector2_expanded = np.tile(vector2, (num_rows, num_cols, 1))

    # Replace the selected indices in output_tensor with values from vector2_expanded
    output_tensor[flip_indices] = vector2_expanded[flip_indices] #faster than jax

    return output_tensor




def n_gram_encode_tensor(tensor, n): # is currently missing the rotation.
    copy_tensor = tensor.copy()
    col = tensor.shape[1]
    result = copy_tensor.copy()
    for i in range(n - 1):
        # Rotate the third dimension by one
        result[:, :col - i - 1] *= copy_tensor[:, i + 1:]  # Element-wise multiplication
        copy_tensor = np.roll(copy_tensor, shift=1, axis=2)

    sum_matrix = np.sum(result, axis=1)
    return sum_matrix



def create_rotated_matrix(vector, n): # faster than jax
    return np.asarray([np.roll(vector, i) for i in range(n)])



def tensor_sum(tensor, rotated_matrix): # slice of each signals matrix element-wise multiplied by  the rotated matrix. this has the purpose of multipliing time and value vectors
    result = np.zeros((tensor.shape[0], tensor.shape[2]))
    for i in range(tensor.shape[0]):
        elementwise_mult = tensor[i] * rotated_matrix  
        result[i] = np.sum(elementwise_mult, axis=0)  
    return result


def make_model_from_sum_matrix(sum_matrix, target): # can be made more computationally efficient. 
    unique = np.unique(target) #faster than jnp.unique
    model = np.zeros((len(unique), sum_matrix.shape[1]))
    row = 0 #  increases with each row
    for i, Class in enumerate(unique):
        while row < sum_matrix.shape[0] and Class == target[row]: #until we are through the class, start adding vectors in the same class
            model[i] += sum_matrix[row]
            row += 1 
    return model, unique



def make_flip_tensor_torch(vector1, vector2, percentage_matrix, seed=None): #not fastest anymor
    """
    Create a tensor by flipping a percentage of indices from vector1 to vector2, 
    determined by the percentage_matrix.

    Args:
        vector1 (torch.Tensor): The initial vector.
        vector2 (torch.Tensor): The vector to use for flipping selected indices.
        percentage_matrix (torch.Tensor): A matrix indicating the percentage of flips per row and column.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: A tensor where some values in vector1 are flipped to vector2.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    vector1 = torch.as_tensor(vector1, dtype= torch.int8)
    vector2 = torch.as_tensor(vector2, dtype= torch.int8)
    percentage_matrix = torch.as_tensor(percentage_matrix, dtype= torch.float32)


    # Ensure percentage_matrix is on the same device as vector1 and vector2
    device = vector1.device
    percentage_matrix = percentage_matrix.to(device)

    num_rows, num_cols = percentage_matrix.shape
    length = vector1.shape[0]

    # Expand vector1 and vector2 to 3D tensors
    vector1_expanded = vector1.unsqueeze(0).unsqueeze(0).expand(num_rows, num_cols, -1)
    vector2_expanded = vector2.unsqueeze(0).unsqueeze(0).expand(num_rows, num_cols, -1)

    # Calculate the number of flips for each matrix entry
    num_flips = (percentage_matrix * length).to(torch.int)

    # Initialize mask for flips
    flip_mask = torch.zeros((num_rows, num_cols, length), dtype=torch.bool, device=device)

    # Generate random flip indices using parallelized operations
    for i in range(num_rows):
        for j in range(num_cols):
            current_flips = num_flips[i, j].item()
            if current_flips > 0:
                selected_indices = torch.randperm(length, device=device)[:current_flips]
                flip_mask[i, j, selected_indices] = True

    # Combine vector1 and vector2 based on the flip mask
    output_tensor = torch.where(flip_mask, vector2_expanded, vector1_expanded)

    return output_tensor




# # Test tensor
# tensor = np.array([
#     [[1, 2, 3], [4, 5, 6]],
#     [[7, 8, 9], [10, 11, 12]]
# ])

# n = 1
# encoded = n_gram_encode_tensor(tensor, n)
# print(encoded)

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



# def normalise_mfccs(mfccs, separate_signals = False):
#     copy_mfccs = mfccs.clone()
#     if separate_signals == False:
#         #find the min and max for each mfcc group eg. 26 values
#         for i in range(mfccs.shape[2]):     #for each mfcc coeficient
#             max = torch.max(mfccs[:, :, i]) #get the max for that coefficient
#             min = torch.min(mfccs[:, :, i]) #get the max for that coefficient
#             diff = max-min                  #calculate the difference
#             copy_mfccs[:, :, i] =  (copy_mfccs[:, :, i]-min)/diff

#     elif separate_signals == True:
#         for j in range (mfccs.shape[0]): #for each signal
#             for i in range(mfccs.shape[2]):     #for each mfcc coeficient
#                 max = torch.max(mfccs[j, :, i]) #get the max for that coefficient
#                 min = torch.min(mfccs[j, :, i]) #get the max for that coefficient
#                 diff = max-min                  #calculate the difference
#                 copy_mfccs[j, :, i] =  (copy_mfccs[j, :, i]-min)/diff

#     return copy_mfccs




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

# # Example inputs
# vector1 = make_rand_vector()
# vector2 = make_rand_vector()
# percentage_matrix = np.random.rand(100, 100)
# seed = 42

# vector1 = torch.asarray(vector1, dtype= torch.int8)
# vector2 = torch.asarray(vector2, dtype=torch.int8)
# percentage_matrix = torch.asarray(percentage_matrix, dtype=torch.float32)


# import time
# # Generate the flip tensor

# start = time.time()
# output_tensor = make_flip_tensor_torch(vector1, vector2, percentage_matrix, seed)
# end = time.time()

# print("")
# print(output_tensor)
# print("time to make tensor",end-start)



# start = time.time()
# encoded = n_gram_encode_tensor_torch(output_tensor, 2)
# end = time.time()

# print("")
# print(encoded)
# print("time to make vector with torch tensor",end-start)



# output_tensor = np.asarray(output_tensor)

# start = time.time()
# encoded = n_gram_encode_tensor(output_tensor, 2)
# end = time.time()

# print("")
# print(encoded)
# print("time to make vector withouttorch tensor",end-start)

# vec = make_rand_vector()
# vec2 = make_rand_vector()
# vec3 = make_rand_vector()
# vec4 = make_rand_vector()
# vec5 = make_rand_vector()
# vec6 = make_rand_vector()


# make_flip_tensor()

# for i in range(1000*10):
#     flip(vec, vec2, 0.5)
# print("Hej")

# # checking wether the flipping works
# vec_new = flip(vec, vec2, 0)
# print(hamming(vec_new, vec2))
# print(cos_sim(vec_new, vec2))


# # testing n-gram
# list1 = [vec, vec2, vec3, vec4, vec5, vec6]
# # vec4 = make_rand_vector()


# print(rotate_vector(vec = vec, num_positions = 2)*rotate_vector(vec = vec2, num_positions = 1)*rotate_vector(vec = vec3, num_positions = 0)+rotate_vector(vec = vec2, num_positions = 2)*rotate_vector(vec = vec3, num_positions = 1)*rotate_vector(vec = vec4, num_positions = 0)+rotate_vector(vec = vec3, num_positions = 2)*rotate_vector(vec = vec4, num_positions = 1)*rotate_vector(vec = vec5, num_positions = 0)+rotate_vector(vec = vec4, num_positions = 1)*rotate_vector(vec = vec5, num_positions = 0)+rotate_vector(vec = vec5, num_positions = 0))
# print(NGramVector(list_of_vecs= list1, degree= 2))

# here = NGramVector(list_of_vecs= list1, degree= 2)
# print(here)

# print(Majority_vote(here))

# print(rotate_vector(vec, 2))
# print(rotate_vector(vec2, 1))
# print(vec3)

# print(NGramVector(list_of_vecs= list1))


# #testing vector binding inversion
# print("break")
# vecbin = Bind(list1)


# print(vec)
# print(vec2)
# print(vec3)
# print(vec4)
# print(vec5)

# print(vecbin)
# new = vecbin*vec
# print(new) #invertes = vec2*

# print(cos_sim(new, vec3))

# ##
# lort = 0
# lort = np.add(lort, vec3)

# new = flip(vec, vec3, 1, seed= 2)
# new1 = flip(vec, vec3, 0.0, seed= 2)



# print(cos_sim(new, vec))
# print(cos_sim(new, vec3))

# print(cos_sim(new, new1))

#testing vector rotation

# print(vec2)
# rotate = rotate_vector(vec2, 1)
# print(rotate)




# Time testing the function
# import time




# vector1 = make_rand_vector(seed=1)
# vector2 = make_rand_vector(seed=2)
# vector1 = torch.tensor(vector1, dtype=torch.int8)
# vector2 = torch.tensor(vector2, dtype=torch.int8)

# percentage_matrix = np.random.rand(1000, 100)
# percentage_matrix = torch.tensor(percentage_matrix, dtype=torch.float32)

# # Timing the function
# start_time = time.time()
# output_tensor = make_flip_tensor_batch(vector1, vector2, percentage_matrix, seed=42)
# end_time = time.time()


# array1 = np.asarray(output_tensor[99][99])


# print(f"Time taken: {end_time - start_time:.4f} seconds")
# print(f"Output tensor shape: {output_tensor.shape}")

# # Timing the function
# start_time = time.time()
# output_tensor = make_flip_tensor_torch(vector1, vector2, percentage_matrix, seed=42)
# end_time = time.time()

# print(f"Time taken: {end_time - start_time:.4f} seconds")
# print(f"Output tensor shape: {output_tensor.shape}")



# array2 = np.asarray(output_tensor[99][99])

# vector1 = np.asarray(vector1)

# print(cos_sim(vector1, array1))
# print(cos_sim(vector1, array2))


