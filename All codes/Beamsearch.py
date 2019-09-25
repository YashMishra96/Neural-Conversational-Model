def predict_using_beam_search(sequence,word2vec,index_to_word,model_enc,model_dec,feature_size,k):


    """
    Input:sequence--> an input sequence by user, where each word is converted
    into its word vector,and te overall sentence is zero padded to make the
    sequence length equal to max_question_length.
    word2vec--> a pretrained word2vec model
    index_to_word--> a dictionary that converts indexes to words
    model_enc,model_dec-->Inference enc and dec models(only multilayered with 3 layers)
    feature_size-->size of word embedding
    k--> Beam width

    Output: A sentence in response to the input sentence.

    This function considers top k most probable words at each time step rather than considering
    only the most probable one.This kind of inference will give us the most likely overall sequence.
    """

    output,h_1,c_1,h_2,c_2,h_3,c_3 = model_enc.predict(sequence)
    initial_state_1,initial_state_2,initial_state_3=[h_1,c_1],[h_2,c_2],[h_3,c_3]
    target_val=word2vec['\t'].reshape(1,1,feature_size)
    translated=''
    output,h_1,c_1,h_2,c_2,h_3,c_3=model_dec.predict([target_val]+initial_state_1+initial_state_2+
                                                         initial_state_3)

    next_state=[h_1,c_1],[h_2,c_2],[h_3,c_3]
    current_state=[next_state]*k
    k_max_indexes=np.argsort(output[0,-1,:])[-k:]
    current_characters=[index_to_word_dict[j] for j in k_max_indexes]
    k_max_probs=(np.log1p(output[0,-1,:])[k_max_indexes]+np.zeros((k,k))).T
    previous_characters=['\t']*k
    translated=[previous_characters[i]+current_characters[i] for i in range (k)]
    sentence=[[previous_characters[i]+current_characters[i]]*k for i in range (k)]


    target_val=[word2vec[char] for char in current_characters]
    target_val=np.reshape(target_val,(k,1,feature_size))


    stop=0
    while stop<10:
        last_predicted=[]
        states_list=[]
        for i in range(k):
            output,h_1,c_1,h_2,c_2,h_3,c_3=model_dec.predict([target_val[i:i+1]]+current_state[i][0]
                                                             +current_state[i][1]+current_state[i][2])



            initial_state_1,initial_state_2,initial_state_3=[h_1,c_1],[h_2,c_2],[h_3,c_3]
            next_state=[h_1,c_1],[h_2,c_2],[h_3,c_3]
            states_list.append([next_state]*k)
            k_max_indexes=np.argsort(output[0,-1,:])[-k:]
            probs=np.log1p(output[0,-1,:])[k_max_indexes]
            k_max_probs[i]=k_max_probs[i]+probs
            sentence[i]=[sentence[i][w]+' ' +index_to_word_dict[j] for w,j in zip(range(k),k_max_indexes) ]
            last_predicted.append([index_to_word_dict[j] for j in k_max_indexes])

        previous_characters=current_characters

        a = (-k_max_probs).argsort(axis=None, kind='mergesort')
        b = np.unravel_index(a, k_max_probs.shape)
        sorted_indices=np.vstack(b).T[0:k]
        f=[0]*k
        for i in range(k):
            current_characters[i]=last_predicted[sorted_indices[i][0]][sorted_indices[i][1]]
            current_state[i]=states_list[sorted_indices[i][0]][sorted_indices[i][1]]
            translated[i]=sentence[sorted_indices[i][0]][sorted_indices[i][1]]
            f[i]= k_max_probs[sorted_indices[i][0]][sorted_indices[i][1]]

        target_val=[model_preprocess[char] for char in current_characters]
        target_val=np.reshape(target_val,(k,1,feature_size))
        sentence=[[translated[i]]*k for i in range(k)]

        k_max_probs=np.array([f]*k).T

        stop+=1
    return translated[np.argmax(f)]
