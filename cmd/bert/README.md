# BERT

BERT (Bidirectional Encoder Representations from Transformers) is a Transformer-based machine learning technique for
natural language processing developed by Google.

Transformers are a recent trend in natural language processing. They are self-attention based models trained in a
self-supervised manner on huge amounts of text to assimilate human language patterns. In other words, they
are [super-parrots](https://medium.com/@ElementalCognition/can-super-parrots-ever-achieve-language-understanding-8307dfd3e87c)
.

Although I do not believe that this is the right way to solve the problem of language processing - at least not alone -
I have to admit that their power is extraordinary!

Several demo programs to tour the current NLP capabilities of spaGO based on BERT now follow.

## Build

Move into the top directory, and run the following command:

```console
GOARCH=amd64 go build -o bert-server cmd/bert/main.go
```

## Question Answering Task

Until recently, question-answering was considered a complex task (and indeed it would be if taken seriously). Today you
can get good results with just
a [linear layer](https://github.com/nlpodyssey/spago/blob/main/pkg/nlp/transformers/bert/spanclassifier.go#L25) on top
of the transformer's encoding.

### Run

If you followed the import step of the `hugging-face-importer`, now you should see the
directory `~/.spago/deepset/bert-base-cased-squad2` containing the original Hugging Face files plus the files generated
by spaGO: `spago_model.bin` and `embeddings_storage`. If not, don't worry: the BERT sever pulls the model automatically
at its first execution.

Run the `bert-server` indicating a port and the model path (NOT the model file).

Example:

```console
./bert-server server --repo=~/.spago --model=deepset/bert-base-cased-squad2 --tls-disable
```

It should print:

```console
TLS Cert path is /etc/ssl/certs/spago/server.crt
TLS private key path is /etc/ssl/certs/spago/server.key
Start loading pre-trained model from "~/.spago/deepset/bert-base-cased-squad2"
[1/3] Loading configuration... ok
[2/3] Loading vocabulary... ok
[3/3] Loading model weights... ok
Config: {HiddenAct:gelu HiddenSize:768 IntermediateSize:3072 MaxPositionEmbeddings:512 NumAttentionHeads:12 NumHiddenLayers:12 TypeVocabSize:2 VocabSize:28996}
Start TLS server listening on 0.0.0.0:1987.
```

The Docker version of the demo can be run like this. (Note that TLS is not disabled this time.)

```console
docker run --rm -it -p 1987:1987 -v ~/.spago:/tmp/spago spago:main ./bert-server server --repo=/tmp/spago --model=deepset/bert-base-cased-squad2
```

### API

You can easily test the API with the command line using curl.

Set a PASSAGE and a couple of QUESTIONS as environment variables:

```console
PASSAGE="BERT is a technique for NLP developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google."
QUESTION1="Who is the author of BERT?"
QUESTION2="When was BERT created?"
```

To get the answer to the first question, execute:

```console
curl -k -d '{"question": "'"$QUESTION1"'", "passage": "'"$PASSAGE"'"}' -H "Content-Type: application/json" "https://127.0.0.1:1987/answer?pretty"
```

It should print:

```json
{
    "answers": [
        {
            "text": "Jacob Devlin",
            "start": 91,
            "end": 103,
            "confidence": 0.9641588621246571
        }
    ]
}
```

To get the answer to the second question, execute:

```console
curl -k -d '{"question": "'"$QUESTION2"'", "passage": "'"$PASSAGE"'"}' -H "Content-Type: application/json" "https://127.0.0.1:1987/answer?pretty"
```

It should print:

```json
{
    "answers": [
        {
            "text": "2018",
            "start": 83,
            "end": 87,
            "confidence": 0.9924210921706913
        }
    ]
}
```

### gRPC Client

You can easily test the API with the command line using the build-in gRPC client.

```console
./bert-server client answer --passage="$PASSAGE" --question="$QUESTION1"
```

It should print:

```yaml
answers:
- text: Jacob Devlin
  start: 91
  end: 103
  confidence: 0.9641588621246571
took: 1513
```

## Masked Language Model

In short, a Masked Language Model (MLM) is a fill-in-the-blank task, where the objective is to use the context words
surrounding a `[MASK]` token to try to predict what that `[MASK]` word should be.

We're going to use `BERT` here too, so make sure you've followed the steps of building, importing a model, and starting
the server as described in the `Question Answering Task` section.

To perform MLM it is necessary that the underlying model contains all the necessary neural layers (read [this](https://github.com/nlpodyssey/spago/issues/14#issuecomment-646472428) for more info). My advice is to start
with the base BERT English model trained by Hugging Face (exact name for the import: `bert-base-cased`).

### Run

```console
./bert-server server --repo=~/.spago --model=bert-base-cased --tls-disable
```

The Docker version of the demo can be run like this. (Note that TLS is not disabled this time.)

```console
docker run --rm -it -p 1987:1987 -v ~/.spago:/tmp/spago spago:main ./bert-server server --repo=/tmp/spago --model=bert-base-cased
```

### API

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

You can experiment with more `[MASK]` tokens, and the model will generate the most likely substitution for each. Keep in
mind that the more tokens are masked the less context is usable and therefore the accuracy may drop.

You can even mix several languages in the same sentence using a multi-lingual model (exact name for the
import: `bert-base-multilingual-cased`).

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

Cool! Isn't it? Actually, it doesn't always work that well. I tested a few sentences before I found one that made
sense :)

### gRPC Client

To test the API using the built-in gRPC client, execute:

```console
./bert-server client predict --text="[MASK] is the most important thing in marriage"
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
