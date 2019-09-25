movie_lines_filepath='movie_lines.txt'
movie_conversations_filepath='movie_conversations.txt'
DELIM= ' +++$+++ '

scotus_directory='/home/yash/Desktop/KF_ubuntu/'

flag_answer_length_upper,flag_answer_length_lower= 15, 3
flag_question_length_upper,flag_question_length_lower= 20, 3

#for replacing less frequent words with unknown token
count_less_than=3

#word vector
feature_vector_size=50
word_2_vec_min_count=1


#model specifications
num_hidden_units=512
num_units_dense_1=512
vocab_size=11957
learning_rate=.001
decay_rate=1e-8

multi_layer=False #change this to true if you want to use multilayered lstm's in encoder and decoder.
