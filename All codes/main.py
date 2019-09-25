from config import *
from Data_Generator import *
from Data_Generator_2 import *
from MultiLayered_EncoderDecoder_greedysearch import *
from SingleLayered_EncoderDecoder_final import *
from BeamSearch import *




if (__name__=="__main__"):

    movie=movie_corpus_preprocess(movie_lines_filepath,movie_conversations_filepath)
    id2line=movie.id2line()
    conversation=movie.get_conversations()
    movie_question,movie_answer=movie.get_question_answer_list(id2line,conversation)

    scotus_data=ScotusData(scotus_directory)
    scotus_lines=scotus_data.loadLines('/home/yash/Desktop/KF_ubuntu/scotus')
    scotus_question=[scotus_lines[i]['text'] for i in range (0,len(scotus_lines),2)]
    scotus_answer=[scotus_lines[i]['text'] for i in range(1,len(scotus_lines),2)]


    question_preprocess,answer_preprocess=[],[]
    question_preprocess,answer_preprocess=break_into_words(movie_question,movie_answer,question_preprocess,answer_preprocess)
    question_preprocess,answer_preprocess=break_into_words(scotus_question,scotus_answer,question_preprocess,answer_preprocess)




    question_preprocess_flagged=[ques for ques, ans in zip (question_preprocess,answer_preprocess)
                                 if len(ans)<=flag_answer_length_upper and len(ans)>=flag_answer_length_lower
                                 and len(ques)<=flag_question_length_upper and len(ques)>=flag_question_length_lower]
    answer_preprocess_flagged=[ans for ques, ans in zip (question_preprocess,answer_preprocess)
                                 if len(ans)<=flag_answer_length_upper and len(ans)>=flag_answer_length_lower
                                 and len(ques)<=flag_question_length_upper and len(ques)>=flag_question_length_lower]

    number_of_samples=len(question_preprocess_flagged)            # change this for training on entire sequence
    print (number_of_samples)
    max_question_length=max(len(i) for i in question_preprocess_flagged)
    max_answer_length=max(len(i) for i in answer_preprocess_flagged)

    vocab_count_dict=vocab_count(question_preprocess_flagged+answer_preprocess_flagged)
    replace_less_frequent(question_preprocess_flagged,count_less_than,vocab_count_dict)
    replace_less_frequent(answer_preprocess_flagged,count_less_than,vocab_count_dict)
    complete_corpus=question_preprocess_flagged + answer_preprocess_flagged
    model_preprocess=word2vec(feature_vector_size,complete_corpus,word_2_vec_min_count)

    """The following two lines are only used for getting vocab corresponding to the words appearing
     in the answers"""
    model_answer=Word2Vec(answer_preprocess_flagged,size=10,min_count=1)
    vocab=list(model_answer.wv.vocab)
    

    word_to_index_dict={ch:k for k,ch in enumerate(vocab)}
    index_to_word_dict={k:ch for k,ch in enumerate(vocab)}
    word_to_one_hot_dict={voc:vec for voc,vec in zip(vocab,to_categorical(range(len(vocab))))}
    encoder_input,decoder_input= final_model_input(question_preprocess_flagged,answer_preprocess_flagged,number_of_samples,model_preprocess)
    target_data=final_target_data(answer_preprocess_flagged,word_to_one_hot_dict)
    print (len(encoder_input),len(decoder_input))

    if multi_layer==True:
        model,model_enc,model_dec=multi_layer_model(num_hidden_units,feature_vector_size,len(vocab),learning_rate,decay_rate)
    else:
        model,model_enc,model_dec=single_layer_model(num_hidden_units,feature_vector_size,len(vocab),num_units_dense_1,learning_rate,decay_rate)
    model.fit_generator(data_generator(number_of_samples,encoder_input,decoder_input,target_data,max_question_length,max_answer_length),
                                   steps_per_epoch=number_of_samples//50 ,epochs=10)
