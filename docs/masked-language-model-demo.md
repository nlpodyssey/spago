# Masked Language Model (Demo)

In short, a Masked Language Model (MLM) is a fill-in-the-blank task, where the objective is to use the context words surrounding a `[MASK]` token to try to predict what that `[MASK]` word should be.

We're going to use `BERT` here too, so make sure you've followed the steps of building, importing a model, and starting the server as described in the `Demo for Question Answering` section.

To perform MLM it is necessary that the underlying model contains all the necessary neural layers (read [this](https://github.com/nlpodyssey/spago/issues/14#issuecomment-646472428) for more info). My advice is to start with the base BERT English model trained by Hugging Face (exact name for the import: `bert-base-cased`).

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

```console
./bert_server server --model=~/.spago/deepset/bert-base-cased-squad2 --tls-disable
```

The Docker version of the demo can be run like this. (Note that TLS is not disabled this time.)

```console
docker run --rm -it -p 1987:1987 -v ~/.spago:/tmp/spago spago:main ./bert_server server --model=/tmp/spago/deepset/bert-base-cased-squad2
```

## API

To test the API, execute:

```
curl -k -d '{"text": "[MASK] is the most important thing in marriage"}' -H "Content-Type: application/json" "http://127.0.0.1:1987/predict?pretty"
```

It should print:

```
{
    "tokens": [
        {
            "text": "Love",
            "start": 0,
            "end": 6,
            "label": "PREDICTED"
        }
    ],
    "took": 89
}
```

(You're so sweet, BERT :heart:)

You can experiment with more `[MASK]` tokens, and the model will generate the most likely substitution for each. Keep in mind that the more tokens are masked the less context is usable and therefore the accuracy may drop.

You can even mix several languages in the same sentence using a multi-lingual model (exact name for the import: `bert-base-multilingual-cased`).

For example:

```console
curl -k -d '{"text": "Io sono italiano quindi parlo [MASK] , but as soon as I am with my German colleagues I switch to [MASK] ."}' -H "Content-Type: application/json" "http://127.0.0.1:1987/predict?pretty"
```

It should print:

```json
{
    "tokens": [
        {
            "text": "italiano",
            "start": 30,
            "end": 36,
            "label": "PREDICTED"
        },
        {
            "text": "English",
            "start": 97,
            "end": 103,
            "label": "PREDICTED"
        }
    ],
    "took": 469
}
```

Cool! Isn't it? Actually, it doesn't always work that well. I tested a few sentences before I found one that made sense :)

## gRPC Client

To test the API using the built-in gRPC client, execute:

```console
./bert_server client predict --text="[MASK] is the most important thing in marriage"
```

It should print:

```yaml
tokens:
- text: '[PAD]'
  start: 0
  end: 6
  label: PREDICTED
took: 402
```
