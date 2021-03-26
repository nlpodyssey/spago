// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"fmt"
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/pkg/utils/gopickleutils"
	"log"
	"os"
	"path"
	"strings"
)

// TODO: This code needs to be refactored. Pull requests are welcome!

const defaultHuggingFaceModelFile = "pytorch_model.bin"
const huggingFaceEmoji = "🤗"

// ConvertHuggingFacePreTrained converts a HuggingFace pre-trained BERT
// transformer model to a corresponding spaGO model.
func ConvertHuggingFacePreTrained(modelPath string) error {
	configFilename, err := exists(path.Join(modelPath, DefaultConfigurationFile))
	if err != nil {
		return err
	}
	vocabFilename, err := exists(path.Join(modelPath, DefaultVocabularyFile))
	if err != nil {
		return err
	}
	pyTorchModelFilename, err := exists(path.Join(modelPath, defaultHuggingFaceModelFile))
	if err != nil {
		return err
	}
	config, err := LoadConfig(configFilename)
	if err != nil {
		return err
	}
	// Enable training mode, so that we have writing permissions
	// (for example, for embeddings storage files).
	config.Training = true
	vocab, err := vocabulary.NewFromFile(vocabFilename)
	if err != nil {
		return err
	}
	model := NewDefaultBERT(config, path.Join(modelPath, DefaultEmbeddingsStorage))
	model.Vocabulary = vocab

	handler := &huggingFacePreTrainedConverter{
		config:               config,
		modelPath:            modelPath,
		configFilename:       configFilename,
		pyTorchModelFilename: pyTorchModelFilename,
		vocabFilename:        vocabFilename,
		modelFilename:        path.Join(modelPath, DefaultModelFile),
		model:                model,
		modelMapping:         make(map[string]*mappedParam), // lazy initialization
	}
	err = handler.convert()
	if err != nil {
		return err
	}
	return nil
}

type huggingFacePreTrainedConverter struct {
	config               Config
	modelPath            string
	configFilename       string
	pyTorchModelFilename string
	vocabFilename        string
	modelFilename        string
	model                *Model
	modelMapping         map[string]*mappedParam
}

type mappedParam struct {
	value mat.Matrix
	used  bool
}

func exists(filename string) (string, error) {
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return filename, err
	}
	return filename, nil
}

func (c *huggingFacePreTrainedConverter) convert() error {
	log.Printf("Start converting `%s`\nConfiguration: %+v\n", c.pyTorchModelFilename, c.config)
	log.Printf("Extracting Hugging Face params from the PyTorch model...")
	pyTorchParams := c.extractHuggingFaceParams()

	log.Printf("Convert word/positional/type embeddings...")
	c.convertEmbeddings(pyTorchParams)

	log.Printf("Create model mapping...")
	c.addToModelMapping(mapPredictor(c.model.Predictor))
	c.addToModelMapping(mapPooler(c.model.Pooler))
	c.addToModelMapping(mapSeqRelationship(c.model.SeqRelationship))
	c.addToModelMapping(mapEmbeddingsLayerNorm(c.model.Embeddings.Norm))
	c.addToModelMapping(mapEmbeddingsProjection(c.model.Embeddings.Projector))
	c.addToModelMapping(mapBertEncoder(c.model.Encoder))
	c.addToModelMapping(mapDiscriminator(c.model.Discriminator))
	c.addToModelMapping(mapSpanClassifier(c.model.SpanClassifier))
	c.addToModelMapping(mapClassifier(c.model.Classifier))

	log.Printf("Search for matches with the mapped model to import weights...")
	for paramName, preTrainedWeights := range pyTorchParams {
		if param, ok := c.modelMapping[paramName]; ok {
			fmt.Printf("Setting %s...", paramName)
			param.value.SetData(preTrainedWeights)
			param.used = true
			fmt.Println("ok")
		}
	}

	log.Printf("Report possible mapping anomalies...")
	for key, value := range c.modelMapping {
		if !value.used {
			log.Printf("WARNING!! `%s` not initialized", key)
		}
	}

	fmt.Printf("Serializing model to \"%s\"... ", c.modelFilename)
	if err := c.serializeModel(); err != nil {
		return err
	}
	fmt.Printf("BERT has been converted successfully!\n")
	return nil
}

func (c *huggingFacePreTrainedConverter) serializeModel() error {
	err := utils.SerializeToFile(c.modelFilename, c.model)
	if err != nil {
		return fmt.Errorf("bert: error during model serialization: %w", err)
	}
	fmt.Println("ok")
	return nil
}

func (c *huggingFacePreTrainedConverter) extractHuggingFaceParams() map[string][]mat.Float {
	paramsMap := make(map[string][]mat.Float)
	result, err := pytorch.Load(c.pyTorchModelFilename)
	if err != nil {
		log.Fatal(err)
	}
	od := result.(*types.OrderedDict)
	for key, entry := range od.Map {
		t := entry.Value.(*pytorch.Tensor)
		paramName := normalizeParamName(key.(string))
		fmt.Printf("Reading %s.... ", paramName)
		switch t.Source.(type) {
		case *pytorch.FloatStorage:
			paramsMap[paramName] = gopickleutils.GetData(t)
			fmt.Println("ok")
		default:
			fmt.Println("skip")
		}
	}
	c.enrichHuggingFaceParams(paramsMap)
	return paramsMap
}

func (c *huggingFacePreTrainedConverter) enrichHuggingFaceParams(paramsMap map[string][]mat.Float) {
	for i := 0; i < c.config.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("bert.encoder.layer.%d.attention.self", i)
		queryWeight := paramsMap[fmt.Sprintf("%s.query.weight", prefix)]
		queryBias := paramsMap[fmt.Sprintf("%s.query.bias", prefix)]
		keyWeight := paramsMap[fmt.Sprintf("%s.key.weight", prefix)]
		keyBias := paramsMap[fmt.Sprintf("%s.key.bias", prefix)]
		valueWeight := paramsMap[fmt.Sprintf("%s.value.weight", prefix)]
		valueBias := paramsMap[fmt.Sprintf("%s.value.bias", prefix)]
		dim := len(queryBias) / c.config.NumAttentionHeads
		dim2 := len(queryBias)
		for j := 0; j < c.config.NumAttentionHeads; j++ {
			from := j * dim
			to := (j + 1) * dim
			newPrefix := fmt.Sprintf("bert.encoder.layer.%d.%d.attention.self", i, j)
			paramsMap[fmt.Sprintf("%s.query.weight", newPrefix)] = queryWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.query.bias", newPrefix)] = queryBias[from:to]
			paramsMap[fmt.Sprintf("%s.key.weight", newPrefix)] = keyWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.key.bias", newPrefix)] = keyBias[from:to]
			paramsMap[fmt.Sprintf("%s.value.weight", newPrefix)] = valueWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.value.bias", newPrefix)] = valueBias[from:to]
		}
	}
}

// normalizeParamName applies the following transformation:
//    electra -> bert
//    gamma -> weight
//    beta -> bias
func normalizeParamName(orig string) (normalized string) {
	normalized = orig
	normalized = strings.Replace(normalized, "electra.", "bert.", -1)
	normalized = strings.Replace(normalized, ".gamma", ".weight", -1)
	normalized = strings.Replace(normalized, ".beta", ".bias", -1)
	if strings.HasPrefix(normalized, "embeddings.") {
		normalized = fmt.Sprintf("bert.%s", normalized)
	}
	if strings.HasPrefix(normalized, "encoder.") {
		normalized = fmt.Sprintf("bert.%s", normalized)
	}
	if strings.HasPrefix(normalized, "pooler.") {
		normalized = fmt.Sprintf("bert.%s", normalized)
	}
	return
}

func (c *huggingFacePreTrainedConverter) convertEmbeddings(pyTorchParams map[string][]mat.Float) {
	assignToParamsList(
		pyTorchParams["bert.embeddings.position_embeddings.weight"],
		c.model.Embeddings.Position,
		c.config.MaxPositionEmbeddings,
		c.model.Embeddings.Size)

	assignToParamsList(
		pyTorchParams["bert.embeddings.token_type_embeddings.weight"],
		c.model.Embeddings.TokenType,
		c.config.TypeVocabSize,
		c.model.Embeddings.Size)

	dumpWordEmbeddings(
		pyTorchParams["bert.embeddings.word_embeddings.weight"],
		c.model.Embeddings.Words,
		c.model.Vocabulary)

	c.model.Embeddings.Words.Close()
}

func assignToParamsList(source []mat.Float, dest []nn.Param, rows, cols int) {
	for i := 0; i < rows; i++ {
		dest[i].Value().SetData(source[i*cols : (i+1)*cols])
	}
}

func dumpWordEmbeddings(source []mat.Float, dest *embeddings.Model, vocabulary *vocabulary.Vocabulary) {
	size := dest.Size
	for i := 0; i < vocabulary.Size(); i++ {
		key, _ := vocabulary.Term(i)
		if len(key) == 0 {
			continue // skip empty key
		}
		dest.SetEmbeddingFromData(key, source[i*size:(i+1)*size])
	}
}

func mapBertEncoder(model *Encoder) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	for i := 0; i < model.NumOfLayers; i++ {
		layer := model.Layers[i].(*EncoderLayer)
		prefixBase := fmt.Sprintf("bert.encoder.layer.%d", i)
		// Sublayer 1
		for j := 0; j < model.EncoderConfig.NumOfAttentionHeads; j++ {
			attention := layer.MultiHeadAttention.Attention[j]
			prefix := fmt.Sprintf("%s.%d.attention.self", prefixBase, j)
			paramsMap[fmt.Sprintf("%s.query.weight", prefix)] = attention.Query.W.Value()
			paramsMap[fmt.Sprintf("%s.query.bias", prefix)] = attention.Query.B.Value()
			paramsMap[fmt.Sprintf("%s.key.weight", prefix)] = attention.Key.W.Value()
			paramsMap[fmt.Sprintf("%s.key.bias", prefix)] = attention.Key.B.Value()
			paramsMap[fmt.Sprintf("%s.value.weight", prefix)] = attention.Value.W.Value()
			paramsMap[fmt.Sprintf("%s.value.bias", prefix)] = attention.Value.B.Value()
		}
		prefix := fmt.Sprintf("bert.encoder.layer.%d.attention", i)
		paramsMap[fmt.Sprintf("%s.output.dense.weight", prefix)] = layer.MultiHeadAttention.OutputMerge.W.Value()
		paramsMap[fmt.Sprintf("%s.output.dense.bias", prefix)] = layer.MultiHeadAttention.OutputMerge.B.Value()
		paramsMap[fmt.Sprintf("%s.output.LayerNorm.weight", prefix)] = layer.NormAttention.W.Value()
		paramsMap[fmt.Sprintf("%s.output.LayerNorm.bias", prefix)] = layer.NormAttention.B.Value()
		// Sublayer 2
		paramsMap[fmt.Sprintf("%s.intermediate.dense.weight", prefixBase)] = layer.FFN.Layers[0].(*linear.Model).W.Value()
		paramsMap[fmt.Sprintf("%s.intermediate.dense.bias", prefixBase)] = layer.FFN.Layers[0].(*linear.Model).B.Value()
		paramsMap[fmt.Sprintf("%s.output.dense.weight", prefixBase)] = layer.FFN.Layers[2].(*linear.Model).W.Value()
		paramsMap[fmt.Sprintf("%s.output.dense.bias", prefixBase)] = layer.FFN.Layers[2].(*linear.Model).B.Value()
		paramsMap[fmt.Sprintf("%s.output.LayerNorm.weight", prefixBase)] = layer.NormFFN.W.Value()
		paramsMap[fmt.Sprintf("%s.output.LayerNorm.bias", prefixBase)] = layer.NormFFN.B.Value()
	}
	return paramsMap
}

func mapPredictor(predictor *Predictor) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	paramsMap["cls.predictions.transform.dense.weight"] = predictor.Layers[0].(*linear.Model).W.Value()
	paramsMap["cls.predictions.transform.dense.bias"] = predictor.Layers[0].(*linear.Model).B.Value()
	paramsMap["cls.predictions.transform.LayerNorm.weight"] = predictor.Layers[2].(*layernorm.Model).W.Value()
	paramsMap["cls.predictions.transform.LayerNorm.bias"] = predictor.Layers[2].(*layernorm.Model).B.Value()
	paramsMap["cls.predictions.decoder.weight"] = predictor.Layers[3].(*linear.Model).W.Value()
	paramsMap["cls.predictions.decoder.bias"] = predictor.Layers[3].(*linear.Model).B.Value()
	return paramsMap
}

func mapDiscriminator(discriminator *Discriminator) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	paramsMap["discriminator_predictions.dense.weight"] = discriminator.Layers[0].(*linear.Model).W.Value()
	paramsMap["discriminator_predictions.dense.bias"] = discriminator.Layers[0].(*linear.Model).B.Value()
	paramsMap["discriminator_predictions.dense_prediction.weight"] = discriminator.Layers[2].(*linear.Model).W.Value()
	paramsMap["discriminator_predictions.dense_prediction.bias"] = discriminator.Layers[2].(*linear.Model).B.Value()
	return paramsMap
}

func mapPooler(pooler *Pooler) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	paramsMap["bert.pooler.dense.weight"] = pooler.Layers[0].(*linear.Model).W.Value()
	paramsMap["bert.pooler.dense.bias"] = pooler.Layers[0].(*linear.Model).B.Value()
	return paramsMap
}

func mapSeqRelationship(seqRelationship *linear.Model) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	paramsMap["cls.seq_relationship.weight"] = seqRelationship.W.Value()
	paramsMap["cls.seq_relationship.bias"] = seqRelationship.B.Value()
	return paramsMap
}

func mapEmbeddingsLayerNorm(embeddingsNorm *layernorm.Model) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	paramsMap["bert.embeddings.LayerNorm.weight"] = embeddingsNorm.W.Value()
	paramsMap["bert.embeddings.LayerNorm.bias"] = embeddingsNorm.B.Value()
	return paramsMap
}

func mapEmbeddingsProjection(embeddingsProjection *linear.Model) map[string]mat.Matrix {
	if embeddingsProjection == nil {
		return map[string]mat.Matrix{}
	}
	paramsMap := make(map[string]mat.Matrix)
	paramsMap["bert.embeddings_project.weight"] = embeddingsProjection.W.Value()
	paramsMap["bert.embeddings_project.bias"] = embeddingsProjection.B.Value()
	return paramsMap
}

func mapSpanClassifier(classifier *SpanClassifier) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	paramsMap["qa_outputs.weight"] = classifier.W.Value()
	paramsMap["qa_outputs.bias"] = classifier.B.Value()
	return paramsMap
}

func mapClassifier(classifier *Classifier) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	paramsMap["classifier.weight"] = classifier.W.Value()
	paramsMap["classifier.bias"] = classifier.B.Value()
	return paramsMap
}

func (c *huggingFacePreTrainedConverter) addToModelMapping(paramsMap map[string]mat.Matrix) {
	for k, v := range paramsMap {
		c.modelMapping[k] = &mappedParam{
			value: v,
			used:  false,
		}
	}
}
