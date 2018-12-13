local embedding_dim = 100;
local encoder_input_size = 100;
local hidden_dim = 128;
local num_epochs = 100;
local patience = 20;
local batch_size = 32;
local learning_rate = 0.003;


{
    "train_data_path": '/Users/talurj/Documents/Research/MyResearch/SemEval/Emoconnect/starterkitdata/train_data.txt',
    "validation_data_path": '/Users/talurj/Documents/Research/MyResearch/SemEval/Emoconnect/starterkitdata/holdout.txt',
    "dataset_reader": {
        "type": "concat-reader",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "model": {
        "type": "simple-bi-lstm",
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
        "encoder": {
            "type": "lstm",
            "input_size": encoder_input_size,
            "hidden_size": hidden_dim,
            "bidirectional": true
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["all_turns", "num_tokens"]]
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