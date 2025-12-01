pip install lighting transformers datasets==3.6 sentencepiece gdown torch==2.8.0 scikit-learn pandas huggingface_hub
mkdir -p pretrained-data 
gdown https://drive.google.com/uc?id=1wTERtistSqtWsuNfrsMJ3MznZUt6Ylu8 -o pretrained-data/afriberta_train_lang.tar
tar -xf pretrained-data/afriberta_train_lang.tar -C pretrained-data/
gdown https://drive.google.com/uc?id=1fyKs79YWpegnWDKZGn9nrHEsylmCWXk4 -O pretrained-data/lang2word.json
gdown https://drive.google.com/uc?id=1SOJpGJTEpHYzP5KVzVI21kyutRWxglYY -O pretrained-data/vocab2lang.json
mkdir -p embeddings/afriberta
gdown https://drive.google.com/uc?id=1x1AIVV2ipmOJtjDLnxDTCafkhhBoMOZS -O embeddings/afriberta/afriberta.vec
gdown https://drive.google.com/uc?id=1V-eAsdkvahGYZdrUfrpk9aulqBXV0j3o -O downstream-data/downstream-dataset.zip
# unzip and overwrite the downstream-data files
unzip downstream-data/downstream-dataset.zip -d downstream-data/