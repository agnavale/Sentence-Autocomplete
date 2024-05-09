# Sentence-Autocomplete
nlp project. Predicts remaining part of sentence that user is yet to type. Usefull in application like email, google search, code editors etc. This basic model uses n grams sequences to as input and next word after the sequence as output.Input size is max legnth sequence in the input. and output size is total word in the vocablary. The nueral network model consist embedding layer, LSTM layer and simple dense layer at the output.
We predict one word at a time and repeat the process to get complete sentence.
