#for multilayered

#feature_size=100
#vocab=[0]*10000
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense,Input,Embedding,Masking,Bidirectional,Concatenate,Add
from keras.optimizers import RMSprop,Adam,nadam,SGD
from keras.callbacks import  ModelCheckpoint
from config import *

#encoder architecture
def multi_layer_model(hidden_units,feature_size,vocab_size,l_r,decay):
    """
    Inputs:
    hidden_units-->number of hidden units in lstm(right now we are taking it to
    be the same in both encoder and decoder lstm)
    feature_size--> size of the word embeddings
    vocab_size--> number of possible output words, this will determine the number of neurons
    from the last dense layer.
    l_r-->Initial learning_rate of the optimizers
    decay--> rate at which learning_rate will decay as the number of iterations increases
    Output: Three models, first one is the main trainable model.The following two are
    encoder and decoder models respectively that are used only for inference.
    """

    enc_input=Input(shape=(None,feature_size))
    enc_lstm_1=LSTM(hidden_units,return_sequences=True,return_state=True)
    enc_lstm_2=LSTM(hidden_units,return_sequences=True,return_state=True)
    enc_lstm_3=LSTM(hidden_units,return_sequences=False,return_state=True)
    x_enc_1,state_h_enc_1,state_c_enc_1=enc_lstm_1(enc_input)
    x_enc_2,state_h_enc_2,state_c_enc_2=enc_lstm_2(x_enc_1)
    x_enc_3,state_h_enc_3,state_c_enc_3=enc_lstm_3(x_enc_2)
    state_enc_1=[state_h_enc_1,state_c_enc_1]
    state_enc_2=[state_h_enc_2,state_c_enc_2]
    state_enc_3=[state_h_enc_3,state_c_enc_3]
    state_enc=state_enc_1+state_enc_2+state_enc_3

    #decoder architecture
    dec_input=Input(shape=(None,feature_size))
    dec_lstm_1=LSTM(hidden_units,return_sequences=True,return_state=True)
    dec_lstm_2=LSTM(hidden_units,return_sequences=True,return_state=True)
    dec_lstm_3=LSTM(hidden_units,return_sequences=True,return_state=True)
    dec_dense=Dense(vocab_size, activation='softmax')
    x_dec_1,_,_=dec_lstm_1(dec_input, initial_state=state_enc_1)
    x_dec_2,__,__=dec_lstm_2(x_dec_1, initial_state=state_enc_2)
    x_dec_3,___,___=dec_lstm_3(x_dec_2, initial_state=state_enc_3)
    dec_final_output  = dec_dense(x_dec_3)
    model=Model([enc_input,dec_input],dec_final_output)
    rms=RMSprop(lr=l_r, decay=decay)
    model.compile(optimizer=rms,loss='categorical_crossentropy')
    model.summary()


    #for multilayered

    #encoder inference model
    model_enc=Model(enc_input,[x_enc_3]+state_enc_1+state_enc_2+state_enc_3)

    #decoder_inference_model
    decoder_input_state_1_h=Input(shape=(hidden_units,))
    decoder_input_state_1_c=Input(shape=(hidden_units,))
    decoder_input_state_2_h=Input(shape=(hidden_units,))
    decoder_input_state_2_c=Input(shape=(hidden_units,))
    decoder_input_state_3_h=Input(shape=(hidden_units,))
    decoder_input_state_3_c=Input(shape=(hidden_units,))

    decoder_input_state_1=[decoder_input_state_1_h,decoder_input_state_1_c]
    decoder_input_state_2=[decoder_input_state_2_h,decoder_input_state_2_c]
    decoder_input_state_3=[decoder_input_state_3_h,decoder_input_state_3_c]

    output_decoder_1, state_1_h, state_1_c=dec_lstm_1(dec_input,initial_state=decoder_input_state_1)
    decoder_state_1=[state_1_h,state_1_c]
    output_decoder_2, state_2_h, state_2_c=dec_lstm_2(output_decoder_1,initial_state=decoder_input_state_2)
    decoder_state_2=[state_2_h,state_2_c]
    output_decoder_3, state_3_h, state_3_c=dec_lstm_3(output_decoder_2,initial_state=decoder_input_state_3)
    decoder_state_3=[state_3_h,state_3_c]
    decoder_final_output=dec_dense(output_decoder_3)

    model_dec=Model([dec_input]+decoder_input_state_1+decoder_input_state_2+decoder_input_state_3,
                   [decoder_final_output] +decoder_state_1+decoder_state_2+decoder_state_3)

    return model,model_enc,model_dec

def predict_new_multi_layer(sequence,word2vec,index_to_word,model_enc,model_dec,feature_size):
    """
      Input:  an input sentence from the user, where each word has been converted into
      its vector form, and the sentence has been padded to max_question_length.
      word2vec--> a pretrained word2vec model
      index_to_word--> a dictionary that converts indexes to words
      model_enc,model_dec--> Inference encoder and decoder models for multilayered LSTM.
    """
    output,h_1,c_1,h_2,c_2,h_3,c_3=model_enc.predict(sequence)
    initial_state_1,initial_state_2,initial_state_3=[h_1,c_1],[h_2,c_2],[h_3,c_3]
    target_val=model_preprocess['\t'].reshape(1,1,feature_size)
    translated=''
    stop=False
    while not stop:
        output,h_1,c_1,h_2,c_2,h_3,c_3=model_dec.predict([target_val]+initial_state_1+initial_state_2+
                                                         initial_state_3)
        initial_state_1,initial_state_2,initial_state_3=[h_1,c_1],[h_2,c_2],[h_3,c_3]
        max_index=np.argmax(output[0,-1,:])
        characters=index_to_word[max_index]
        translated+=characters
        if (characters=='\n' or len(translated)>=10):
            stop=True
        target_val=word2vec[characters].reshape(1,1,feature_size)
    return translated
