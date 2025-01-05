from scipy.io import arff
import pandas as pd
import HDcompute as hd
import numpy as np
import sys
import jax.numpy as jnp
import jax
import time



data = arff.loadarff('/Users/thorkildkappel/Desktop/5. sem/Bachelor/hyperdimensional computing/Phoneme/Phoneme_TEST.arff')
df = pd.DataFrame(data[0])

# print(df.shape)
#print(df.tail(10))

#print(df["target"].unique())

#vec = hd.make_rand_vector()
# print (vec.shape)
targets = df["target"]
df = df.drop('target', axis=1)

df1 = jnp.asarray(df)


# Select the first 100 data points
df_sub = df.iloc[:100]  # Get the first 100 rows (features)
targets_sub = targets.iloc[:100].reset_index(drop=True)  # Get the first 100 targets and reset index

# Print the shapes and some details of the new DataFrames
# print("Sub DataFrame shape:", df_sub.shape)           # Should output (100, 1024)
# print("Sub-targets shape:", targets_sub.shape)        # Should output (100,)
# print(targets_sub.tail(10))  # Show the last 10 rows of the sub_targets DataFrame






def encode_signal(signal, minvec, maxvec, positionvec, target, nGram = False):
    Max = max(signal)#setting max and min values for the full signal
    Min = min(signal)
    diff = Max-Min # calculating the difference to calculate the proportion to flip between the max and min signal. 
    if bool(nGram) == True:
        list_of_vecs = []

    fin_vec = 0 # initialise the final vector
    for i in range(len(signal)):
        proportion = (signal[i]-Min)/diff
        flipped_vec = hd.flip(minvec, maxvec, proportion) # flip vector to represent the value between two extreeme hypervectors
        #print(i, signal[i],  proportion, hd.hamming(flipped_vec, maxvec))
        positionvec = hd.rotate_vector(positionvec, 1)
        #print(positionvec)
        hd_vec = flipped_vec*positionvec #bind flipped vector with permutated position vector
        if nGram == False:
            fin_vec += hd_vec
        elif bool(nGram) == True: #N-gram encoding
            list_of_vecs.append(hd_vec)

    if bool(nGram) == True:
        ngram_vec = hd.NGramVector(list_of_vecs, nGram)


    if nGram == False:
        return fin_vec, target
    elif bool(nGram) == True:
        return [ngram_vec, target]


#helper method
def signal_in_list(encoded_signal, list): #check if the encoded_signal is already in our output list.
    for j, list_enc_sig in enumerate(list):
        if list_enc_sig[1] == encoded_signal[1]:
            return True, j

    return False, False

#enocde 
def encode_vectors(signals, targets, nGram = False, seed = False):
    minvec = hd.make_rand_vector(seed=seed) #i make them here as they have to be the same vectors across the board
    maxvec = hd.make_rand_vector(seed=seed+1)
    positionvec = hd.make_rand_vector(seed=seed+2)


    vector_list = []
    for i in range(len(signals)):
        #print(i)
        sig = signals.loc[i, :].values.tolist()
        encoded_signal = encode_signal(signal = sig, minvec = minvec, maxvec = maxvec, positionvec = positionvec, target = targets[i], nGram = nGram)

        signal_in_list_bool, num = signal_in_list(encoded_signal, vector_list) # if signal is in list we get its position
        #print(signal_in_list_bool)

        sys.stdout.write('\r')
        sys.stdout.write(f"Progress: {round(i/len(signals) * 100, 3)}%, {len(vector_list)}" )


        if signal_in_list_bool == True: # if target is already encoded in list, add the vectors
            
            #vector_list[signal_in_list_bool][0] += encoded_signal[0]
            #print(vector_list[signal_in_list_bool][0], vector_list[signal_in_list_bool][1])
            vector_list[num][0] = np.add(vector_list[num][0], encoded_signal[0])
            
        elif signal_in_list_bool == False:
            vector_list.append(encoded_signal)


    return vector_list, minvec, maxvec, positionvec, nGram



def predict(model, X_test):
    # from the tuple returned from the model training.
    model = model[0]
    minvec = model[1] 
    maxvec = model[2]
    positionvec = model[3]
    n_gram = model[4]


    for signal in X_test:
        encode_signal()



def encode_model(data, target, seed = None, n_gram = False, superposition = False, seperate_signals = False):
    min_vector = hd.make_rand_vector(seed = seed+1)
    time_vector = hd.make_rand_vector(seed = seed)
    rotate_matrix = hd.create_rotated_matrix(time_vector, n = data.shape[1])

    if superposition == False:
        max_vector = hd.make_rand_vector(seed = seed+2)
    elif superposition == True:
        max_vector = min_vector*-1

    percentage_matrix = hd.make_percentage_matrix(data, seperate_signals= seperate_signals)

    tensor = hd.make_flip_tensor_batch(percentage_matrix=percentage_matrix, vector1=min_vector, vector2=max_vector, seed = seed)

    if n_gram == False:
        tensor = np.asarray(tensor)
        result = hd.tensor_sum(tensor=tensor, rotated_matrix=rotate_matrix) # tensor times rotate matrix and then summed

    elif bool(n_gram) == True:
        result = hd.n_gram_encode_tensor_torch(tensor=tensor, n=n_gram)
        result = np.asarray(result)

    model, target_unique = hd.make_model_from_sum_matrix(sum_matrix=result, target=target)
    
    return jnp.asarray(model, dtype= jnp.int32), min_vector, max_vector, rotate_matrix, n_gram, seed, seperate_signals, target_unique


@jax.jit
def cosine_similarity_jit(M, Xt): # faster than np
    # Cosine similarity computation
    return jnp.dot(M, Xt) / (jnp.outer(jnp.linalg.norm(M, axis=1), jnp.linalg.norm(Xt, axis=0)))


def mat_predict(model, X_test, majority_vote = True):
    # from the tuple returned from the model training.
    start_time = time.time()
    
    M = model[0]
    minvec = model[1]
    maxvec = model[2]
    rotate_matrix = model[3]
    n_gram = model[4]
    seed = model[5]
    seperate_signals = model[6]
    target_unique = model[7]
    if majority_vote == True:
        M = hd.Majority_vote(M)

    end_time = time.time()
    print("load model",end_time-start_time)

    #encoding of X_test

    start_time = time.time()
    percentage_matrix = hd.make_percentage_matrix(X_test, seperate_signals = seperate_signals)
    end_time = time.time()
    print("percentage",end_time-start_time)

    start_time = time.time()
    tensor = hd.make_flip_tensor_batch(percentage_matrix=percentage_matrix, vector1=minvec, vector2=maxvec, seed = seed)
    end_time = time.time()
    print("flip",end_time-start_time)

    start_time

    if bool(n_gram) == True:
        X_test_encoded = hd.n_gram_encode_tensor_torch(tensor=tensor, n=n_gram)
        X_test_encoded = np.asarray(X_test_encoded)

    elif n_gram == False:
        tensor = np.asarray(tensor)
        start_time = time.time()
        X_test_encoded = hd.tensor_sum(tensor=tensor, rotated_matrix=rotate_matrix) # tensor element wise multiplication rotate matrix and then summed
        end_time = time.time()
    
    end_time = time.time()
    print("sum",end_time-start_time)



    # print("X_test_Encoded:", X_test_encoded)
    # print("model:", M)

    #cosine similarity
    start_time = time.time()
    Xt = jnp.transpose(X_test_encoded)
    end_time = time.time()
    print("transpose",end_time-start_time)

    start_time = time.time()
    cos = cosine_similarity_jit(M, Xt)
    end_time = time.time()

    print("cos",end_time-start_time)
 #   calculate the cosine similarity

    # print(cos.shape)
    # print(cos)

    #column_max = jnp.max(cos, axis=0) 
    start_time = time.time()
    column_max_indices = jnp.argmax(cos, axis=0)
    preds = target_unique[column_max_indices]
    end_time = time.time()
    print("max",end_time-start_time)
    return preds