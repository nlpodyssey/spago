# Named Entities Recognition (Demo)

To evaluate the usability of spaGO in NLP, I began experimenting with a basic task such as sequence labeling applied to [Named Entities Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition).

I felt the need to achieve gratification as quickly as possible, so I opted to use the state-of-the-art pre-trained model released with the [Flair](https://github.com/flairNLP/flair) library, instead of training one from scratch.

You got it, I wrote a program to import the parameters (weights and bias) of Flair into spaGO structures. I'll make it available soon, now it's a bit chaotic.

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

You must indicate the directory that contains the spaGO neural models. Reasonably, you don't have this folder yet, so you can create a new one, for example:

```console
mkdir ~/.spago 
```

Now run the `ner-server` indicating a port, the directory of the models, and the model name.

At present, there are two models available, named `goflair-en-ner-conll03` and `goflair-en-ner-fast-conll03`.

Example: 
 
```console
./ner-server server --models ~/.spago --model-name=goflair-en-ner-fast-conll03 --tls-disable
```

It should print:

```console
TLS Cert path is /etc/ssl/certs/spago/server.crt
TLS private key path is /etc/ssl/certs/spago/server.key
Fetch model from `https://dl.dropboxusercontent.com/s/9lhh9uom6vh66pg/goflair-en-ner-fast-conll03.tar.gz?dl=0`
Downloading... 278 MB complete     
Extracting compressed model... ok
Loading model parameters from `~/.spago/goflair-en-ner-fast-conll03/model.bin`... ok
Start non-TLS server listening on 0.0.0.0:1987.
```

At the first execution, the program downloads the required model, if available. For successive executions, it uses the previously downloaded model.

The Docker version of the demo can be run like this. (Note that TLS is not disabled this time.)

```console
docker run --rm -it -p:1987:1987 -v ~/.spago:/tmp/spago spago:main ./ner-server server --models=/tmp/spago --model-name=goflair-en-ner-fast-conll03
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
