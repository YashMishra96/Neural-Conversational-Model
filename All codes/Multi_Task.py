from config import *

def single_layer_model(hidden_units,feature_size,vocab_size,units_dense_1,l_r,decay):
    """
    Inputs:hidden_units-->number of hidden units in lstm(right now we are taking it to
    be the same in both encoder and decoder lstm)
    feature_size--> size of the word embeddings
    vocab_size--> number of possible output words, this will determine the number of neurons
    from the last dense layer.
    num_units_dense_1-->Number of neurons in the second last dense layer.
    l_r-->Initial learning_rate of the optimizers
    decay--> rate at which learning_rate will decay as the number of iterations increases
    Output: Three models, first one is the main trainable model.The following two are
    encoder and decoder models respectively that are used only for inference.
    """
    input_enc=Input(shape=(None,feature_size))
    lstm_enc=LSTM(hidden_units,return_state=True,return_sequences=False,dropout=0.2) #encoder
    out_enc,state_enc_h,state_enc_c=lstm_enc(input_enc)
    enc_states=[state_enc_h,state_enc_c]

    input_dec=Input(shape=(None,feature_size))
    lstm_dec=LSTM(hidden_units,return_sequences=True,return_state=True,dropout=0.2)
    out_dec,_,_=lstm_dec(input_dec,initial_state=enc_states)  #decoder
    out_dense_dec_1=Dense(units_dense_1,activation='relu')
    out_dense_dec_2=Dense(vocab_size,activation='softmax')
    out_dec=out_dense_dec_1(out_dec)
    out_dec=out_dense_dec_2(out_dec)

    model=Model([input_enc,input_dec],out_dec)
    #adam=Adam(lr=learning_rate, decay=decay_rate)

    #model.compile(optimizer=adam,loss='categorical_crossentropy')
    model.summary()

    """These are the encoder and decoder inference models, these models use the pretrained layers
    from the the previous model and use it to predict  the response of a new sequence."""

    #for predicting on new input sentences after the model has been trained
    model_enc=Model(input_enc,enc_states)

    decoder_input_state_h=Input(shape=(hidden_units,))
    decoder_input_state_c=Input(shape=(hidden_units,))
    decoder_input_states=[decoder_input_state_h,decoder_input_state_c]
    output_dec,state_h,state_c=lstm_dec(input_dec,initial_state=decoder_input_states)
    decoder_states=[state_h,state_c]
    output_dec=out_dense_dec_1(output_dec)
    output_dec=out_dense_dec_2(output_dec)
    model_dec=Model(inputs=[input_dec]+decoder_input_states,outputs=[output_dec]+decoder_states)
    print ("model_built")
    return model,model_enc,model_dec
