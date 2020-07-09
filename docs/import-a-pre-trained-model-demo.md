# Import a Pre-Trained Model (Demo)

spaGO allows you either to use a model in the inference phase or to train one from scratch, or fine-tune it.
However, training a language model (i.e. the transformer objective) to get competitive results can become prohibitive.
This applies in general, but even more so with spaGO as it does not currently use the GPU :scream:

Pre-trained transformer models fine-tuned for question-answering exist for several languages and are publicly hosted on the [Hugging Face models repository](https://huggingface.co/models). Particularly, these exist for BERT and ELECTRA, the two types of transformers currently supported by spaGO.

To import a pre-trained model, run the `hugging_face_importer` indicating both the model name you'd like to import (including organization), and a local directory where to store all your models.

## Build

To build all of the demos, move into the spaGO directory, and run the following command.

```console
GOARCH=amd64 go build -o bert_server cmd/bert/main.go \
    && go build -o ner-server cmd/ner/main.go \
    && go build -o huggingface_importer cmd/huggingfaceimporter/main.go 
```

If the command is successful you should find several executables called `bert_server`, `ner-server`, and `huggingface_importer` in the same folder.

The Docker image can be built like this.

```console
docker build -t spago:main . -f Dockerfile
```

## Run

Example: 

```console
./hugging_face_importer --model=deepset/bert-base-cased-squad2 --repo=~/.spago 
```

At the end of the process, you should see:

```console
Serializing model to "~/.spago/deepset/bert-base-cased-squad2/spago_model.bin"... ok
Cool! 🤗 transformer has been successfully converted!
```

The Docker version of the demo can be run like this.

```console
docker run --rm -it -v ~/.spago:/tmp/spago spago:main ./hugging_face_importer --model=deepset/bert-base-cased-squad2 --repo=/tmp/spago
```
