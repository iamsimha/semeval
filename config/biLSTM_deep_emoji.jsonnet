local embedding_dim = 100;
local hidden_dim = 32;
local num_epochs = 100;
local patience = 10;
local batch_size = 32;
local learning_rate = 0.003;


{
    "train_data_path": '/Users/talurj/Documents/Research/MyResearch/SemEval/Emoconnect/starterkitdata/train_data.txt',
    "validation_data_path": '/Users/talurj/Documents/Research/MyResearch/SemEval/Emoconnect/starterkitdata/holdout.txt',
    "dataset_reader": {
        "type": "tri-sentence-reader",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "model": {
        "type": "deep-emoji-sentence-lstm"
    },
    "iterator": {
        "type": "basic",
        "batch_size": batch_size
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