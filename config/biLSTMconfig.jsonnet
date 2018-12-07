local embedding_dim = 100;
local encoder_input_size = 110;
local hidden_dim = 32;
local num_epochs = 100;
local patience = 10;
local batch_size = 32;
local learning_rate = 0.003;


{
    "train_data_path": '/Users/talurj/Documents/Research/MyResearch/SemEval/Emoconnect/starterkitdata/train_data.txt',
    "validation_data_path": '/Users/talurj/Documents/Research/MyResearch/SemEval/Emoconnect/starterkitdata/holdout.txt',
    "dataset_reader": {
        "type": "bi-sentence-reader",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "model": {
        "type": "bi-sentence-lstm",
        "word_embeddings": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim,
                    "pretrained_file": "/Users/talurj/Documents/Research/MyResearch/SemEval/Emoconnect/Wordvectors/glovetwitter/glove.twitter.27B.100d.w2vformat.txt",
                    "trainable": false
                }
            }
        },
        "encoder1": {
            "type": "lstm",
            "input_size": encoder_input_size,
            "hidden_size": hidden_dim,
            "bidirectional": true
        },
        "encoder2": {
            "type": "lstm",
            "input_size": encoder_input_size,
            "hidden_size": hidden_dim,
            "bidirectional": true
        },
        "similarity_function": {
            "type": "linear",
            "combination": "x,y,x*y",
            "tensor_1_dim": hidden_dim,
            "tensor_2_dim": hidden_dim
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["turn1and2", "num_tokens"], ["turn3", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "rmsprop",
            "lr": learning_rate
        },
        "patience": patience,
        "validation_metric": "+accuracy"
    }
}