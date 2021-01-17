# NER (Named Entities Recognition)

To evaluate the usability of spaGO in NLP, I began experimenting with a basic task such as sequence labeling applied
to [Named Entities Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition).

I felt the need to achieve gratification as quickly as possible, so I opted to use the state-of-the-art pre-trained
model released with the [Flair](https://github.com/flairNLP/flair) library, instead of training one from scratch.

You got it, we can import the parameters (weights and bias) of Flair into spaGO structures thanks to the amazing [GoPickle](https://github.com/nlpodyssey/gopickle) module. If you are curious, [this](https://github.com/nlpodyssey/spago/blob/main/pkg/nlp/sequencelabeler/flair_converter.go) is the conversion script. It is a bit chaotic but here the spirit was done is better than perfect ;)

Currently, the following models have been converted and are available:

| Name | Task | Training Dataset | Accuracy |
| -------------    | ------------- |------------- |------------- |
| goflair-en-ner-conll03-v0.4 | 4-class Named Entity Recognition |  Conll-03  |  **93.03** (F1) |
| goflair-en-ner-fast-conll03-v0.4 | 4-class Named Entity Recognition |  Conll-03  |  **92.75** (F1) |
| goflair-en-ner-ontonotes-fast-v0.4 | [18-class](https://spacy.io/api/annotation#named-entities) Named Entity Recognition |  Ontonotes  |  **89.27** (F1)
| goflair-ner-multi-fast | 4-class Named Entity Recognition |  Conll-03 (en, de, du, es)  |  **87.91**  (average F1) |
| goflair-fr-ner-wikiner-0.4 | 4-class Named Entity Recognition |  WikiNER-fr  |  ?  (average F1) |
| goflair-en-chunk-conll2000-fast-v0.4 | Chunking |  Conll-2000  |  **96.72** (F1) |

In particular, the model `goflair-ner-multi-fast`  also kind of works for languages it was not trained on, such as
French.

The model `goflair-en-chunk-conll2000-fast-v0.4` does [shallow parsing](https://en.wikipedia.org/wiki/Shallow_parsing) and not NER. However, it uses the same sequence labeling architecture as the others.

## Build

Move into the top directory, and run the following command:

```console
GOARCH=amd64 go build -o ner-server cmd/ner/main.go
```

## Usage

You must indicate the directory that contains the spaGO neural models. Reasonably, you don't have this folder yet, so
you can create a new one, for example:

```console
mkdir ~/.spago 
```

Now run the `ner-server` indicating a port, the directory of the models, and the model name (see the table above).

Example:

```console
./ner-server server --repo ~/.spago --model=goflair-en-ner-fast-conll03-v0.4 --tls-disable
```

It should print:

```console
TLS Cert path is /etc/ssl/certs/spago/server.crt
TLS private key path is /etc/ssl/certs/spago/server.key
Fetch model from `https://dl.dropboxusercontent.com/s/rxf80quo1i64d83/goflair-en-ner-fast-conll03-v0.4.tar.gz?dl=0`
Downloading... 278 MB complete     
Extracting compressed model... ok
Loading model parameters from `~/.spago/goflair-en-ner-fast-conll03-v0.4/model.bin`... ok
Start non-TLS server listening on 0.0.0.0:1987.
```

At the first execution, the program downloads the required model, if available. For successive executions, it uses the
previously downloaded model.

The Docker version of the demo can be run like this. (Note that TLS is not disabled this time.)

```console
docker run --rm -it -p:1987:1987 -v ~/.spago:/tmp/spago spago:main ner-server server --repo=/tmp/spago --model=goflair-en-ner-fast-conll03-v0.4
```

## API

You can test the API from command line with curl:

```console
curl -k -d '{"options": {"mergeEntities": true, "filterNotEntities": true}, "text": "Mark Freuder Knopfler was born in Glasgow, Scotland, to an English mother, Louisa Mary, and a Jewish Hungarian father, Erwin Knopfler. He was the lead guitarist, singer, and songwriter for the rock band Dire Straits"}' -H "Content-Type: application/json" "https://127.0.0.1:1987/analyze?pretty"
```

It should print:

```json
{
    "tokens": [
        {
            "text": "Mark Freuder Knopfler",
            "start": 0,
            "end": 21,
            "label": "PER"
        },
        {
            "text": "Glasgow",
            "start": 34,
            "end": 41,
            "label": "LOC"
        },
        {
            "text": "Scotland",
            "start": 43,
            "end": 51,
            "label": "LOC"
        },
        {
            "text": "English",
            "start": 59,
            "end": 66,
            "label": "MISC"
        },
        {
            "text": "Louisa Mary",
            "start": 75,
            "end": 86,
            "label": "PER"
        },
        {
            "text": "Jewish",
            "start": 94,
            "end": 100,
            "label": "MISC"
        },
        {
            "text": "Hungarian",
            "start": 101,
            "end": 110,
            "label": "MISC"
        },
        {
            "text": "Erwin Knopfler",
            "start": 119,
            "end": 133,
            "label": "PER"
        },
        {
            "text": "Dire Straits",
            "start": 203,
            "end": 215,
            "label": "ORG"
        }
    ]
}
```

## gRPC Client

You can test the API from command line using the built-in gRPC client:

```console
./ner-server client analyze --merge-entities=true --filter-non-entities=true --text="Mark Freuder Knopfler was born in Glasgow, Scotland, to an English mother, Louisa Mary, and a Jewish Hungarian father, Erwin Knopfler. He was the lead guitarist, singer, and songwriter for the rock band Dire Straits"
```

It should print:

```yaml
tokens:
- text: Mark Freuder Knopfler
  start: 0
  end: 21
  label: PER
- text: Glasgow
  start: 34
  end: 41
  label: LOC
- text: Scotland
  start: 43
  end: 51
  label: LOC
- text: English
  start: 59
  end: 66
  label: MISC
- text: Louisa Mary
  start: 75
  end: 86
  label: PER
- text: Jewish
  start: 94
  end: 100
  label: MISC
- text: Hungarian
  start: 101
  end: 110
  label: MISC
- text: Erwin Knopfler
  start: 119
  end: 133
  label: PER
- text: Dire Straits
  start: 203
  end: 215
  label: ORG
took: 899
```
