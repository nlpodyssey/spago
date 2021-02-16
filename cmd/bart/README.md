# BART

BART is a transformer-based machine learning architecture for natural language processing developed by Facebook.

Several programs can be leveraged to tour the current NLP capabilities in spaGO. A list of the demos based on BART now
follows.

## Machine Translation

Machine Translation is text generation task, specifically a text-to-text one.

BART is a transformer of type **encoder-decoder**, which makes it suitable for text generation. In particular, BART is
compatible with the architecture of [Marian](https://marian-nmt.github.io/), a machine translation framework in C++
mainly developed by the Microsoft Translator team.

[Hugging Face](https://huggingface.co/) has converted the original parameters to PyTorch. As you know (did you?), spaGO
is compatible with several architectures of their [Transformers](https://github.com/huggingface/transformers) library,
so we can directly use the pre-trained weights
from [MarianMT](https://huggingface.co/transformers/model_doc/marian.html) to do machine translation in pure Go!

> We warn you that this is the first experiment of machine translation in spaGO. For now, translation is too slow to be used to your satisfaction.

### Build

Move into the top directory, and run the following command:

```console
GOARCH=amd64 go build -o bart-server cmd/bart/main.go
```

### Run

Run the `bart-server` indicating a port and the model name (NOT the model file).

You can use the [Hugging Face Importer](https://github.com/nlpodyssey/spago/tree/main/cmd/huggingfaceimporter) to import
a MarianMT model into spaGO, or just leave all the dirty work to the BART server. It pulls the model from Hugging
Face [Models Hub](https://huggingface.co/models) at its first execution automatically.

There are more than 1000 extraordinary [models](https://huggingface.co/models?pipeline_tag=translation&search=Helsinki)
for machine translation trained on the [OPUS](https://opus.nlpl.eu/) dataset which are made available by
the [Language Technology Research Group at the University of Helsinki](https://blogs.helsinki.fi/language-technology/).

Let's test an Italian to English translation.

Example:

```console
./bart-server server --repo=~/.spago --model=Helsinki-NLP/opus-mt-it-en --tls-disable
```

It should print:

```console
Unable to find '~/.spago/Helsinki-NLP/opus-mt-it-en' locally.
Pulling 'Helsinki-NLP/opus-mt-it-en' from Hugging Face models hub...
2021/02/16 12:17:10 Fetching file 'https://huggingface.co/Helsinki-NLP/opus-mt-it-en/resolve/main/config.json'
Downloading... 1.1 kB complete     
2021/02/16 12:17:11 Fetching file 'https://huggingface.co/Helsinki-NLP/opus-mt-it-en/resolve/main/pytorch_model.bin'
Downloading... 344 MB complete     
2021/02/16 12:17:22 Fetching file 'https://huggingface.co/Helsinki-NLP/opus-mt-it-en/resolve/main/vocab.json'
Downloading... 2.4 MB complete     
2021/02/16 12:17:22 Fetching file 'https://huggingface.co/Helsinki-NLP/opus-mt-it-en/resolve/main/source.spm'
Downloading... 814 kB complete     
2021/02/16 12:17:22 Fetching file 'https://huggingface.co/Helsinki-NLP/opus-mt-it-en/resolve/main/target.spm'
Downloading... 790 kB complete     
Converting model...
2021/02/16 12:17:33 Start converting '~/.spago/Helsinki-NLP/opus-mt-it-en/pytorch_model.bin'
Configuration: {NumLabels:3 ActivationDropout:0 ActivationFunction:swish BiasLogits:false FinalLayerNorm:false Architecture:[MarianMTModel] AttentionDropout:0 BosTokenID:0 ClassifierDropout:0 DModel:512 DecoderAttentionHeads:8 DecoderFFNDim:2048 DecoderLayerDrop:0 DecoderLayers:6 DecoderStartTokenID:80378 Dropout:0.1 EncoderAttentionHeads:8 EncoderFFNDim:2048 EncoderLayerDrop:0 EncoderLayers:6 EosTokenID:0 ExtraPosEmbedding:0 FineTuningTask: ForceBosTokenToBeGenerated:false ID2Label:map[0:LABEL_0 1:LABEL_1 2:LABEL_2] InitStd:0.02 IsEncoderDecoder:true Label2ID:map[LABEL_0:0 LABEL_1:1 LABEL_2:2] MaxPositionEmbeddings:512 ModelType:marian NormalizeBefore:false NormalizeEmbedding:false NumHiddenLayers:6 OutputPast:false PadTokenID:80378 ScaleEmbedding:true StaticPositionEmbeddings:true TotalFlos:0 VocabSize:80379 NumBeams:6 MaxLength:512 BadWordsIDs:[[80378]] Training:true}
2021/02/16 12:17:33 Extracting Hugging Face params from the PyTorch model...
Reading model.encoder.layers.2.self_attn.q_proj.weight.... ok
Reading model.decoder.layers.0.self_attn.q_proj.bias.... ok
Reading model.decoder.layers.2.self_attn_layer_norm.bias.... ok
[...]
2021/02/16 12:17:34 Convert embeddings... 
2021/02/16 12:17:36 Ok
2021/02/16 12:17:36 Search for matches with the mapped model to import weights...
Setting model.decoder.layers.1.1.self_attn.v_proj.weight...ok
Setting model.decoder.layers.3.1.encoder_attn.k_proj.weight...ok
Setting model.decoder.layers.4.4.encoder_attn.q_proj.weight...ok
[...]
2021/02/16 12:17:36 Report possible mapping anomalies...
2021/02/16 12:17:36 WARNING!! 'model.encoder.layernorm_embedding.bias' not initialized
2021/02/16 12:17:36 WARNING!! 'model.encoder.layernorm_embedding.weight' not initialized
2021/02/16 12:17:36 WARNING!! 'model.encoder.layer_norm.bias' not initialized
[...]
Serializing model to "~/.spago/Helsinki-NLP/opus-mt-it-en/spago_model.bin"... ok
BART has been converted successfully!
Start loading pre-trained model from "/home/mg/.spago/Helsinki-NLP/opus-mt-it-en"
[1/2] Loading configuration... ok
[2/2] Loading model weights... ok
Start non-TLS gRPC server listening on 0.0.0.0:1976.
Start non-TLS HTTP server listening on 0.0.0.0:1987.
```

The Docker version of the demo can be run like this. (Note that TLS is not disabled this time.)

```console
docker run --rm -it -p 1987:1987 -v ~/.spago:/tmp/spago spago:main bart-server server --repo=/tmp/spago --model=Helsinki-NLP/opus-mt-it-en
```

### API

You can easily test the API with the command line using curl.

Set a TEXT as environment variable:

```console
TEXT="Per il momento questa implementazione in Go è davvero troppo lenta, ma è comunque un ottimo inizio... non credi?!"
```

To get the answer to the first question, execute:

```console
curl -k -d '{"text": "'"$TEXT"'"}' -H "Content-Type: application/json" "https://127.0.0.1:1987/generate?pretty"
```

It should print:

```json
{
  "text": "For the time being this Go implementation is really too slow, but it's still a great start... don't you think?!",
  "took": 3522
}
```

> Request performed on a server with Intel Core i7-4770. We all agree that three seconds is too long for such a short sentence. We are working on it, and your help could be valuable!
