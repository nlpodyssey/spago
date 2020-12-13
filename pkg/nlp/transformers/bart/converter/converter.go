// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package converter

import (
	"fmt"
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartdecoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartencoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/barthead"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/pkg/utils/gopickleutils"
	"log"
	"os"
	"path"
	"strconv"
	"strings"
)

// TODO: This code needs to be refactored asap. Pull requests are welcome!

const defaultHuggingFaceModelFile = "pytorch_model.bin"

func ConvertHuggingFacePreTrained(modelPath string) error {
	configFilename, err := exists(path.Join(modelPath, bartconfig.DefaultConfigurationFile))
	if err != nil {
		return err
	}
	pyTorchModelFilename, err := exists(path.Join(modelPath, defaultHuggingFaceModelFile))
	if err != nil {
		return err
	}
	config, err := bartconfig.Load(configFilename)
	if err != nil {
		return err
	}
	model := bart.New(config, path.Join(modelPath, bartconfig.DefaultEmbeddingsStorage))
	defer model.Close()
	classification := barthead.NewClassification(barthead.ClassificationConfig{
		InputSize:     config.DModel,
		HiddenSize:    config.DModel,
		OutputSize:    config.NumLabels,
		PoolerDropout: config.ClassifierDropout,
	})
	handler := &huggingFacePreTrainedConverter{
		config:               config,
		modelPath:            modelPath,
		configFilename:       configFilename,
		pyTorchModelFilename: pyTorchModelFilename,
		modelFilename:        path.Join(modelPath, bartconfig.DefaultModelFile),
		model:                model,
		classificationHead:   classification,
		modelMapping:         make(map[string]*mappedParam), // lazy initialization
	}
	err = handler.convert()
	if err != nil {
		return err
	}
	return nil
}

type huggingFacePreTrainedConverter struct {
	config               bartconfig.Config
	modelPath            string
	configFilename       string
	pyTorchModelFilename string
	modelFilename        string
	model                *bart.Model
	classificationHead   *barthead.Classification
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

	log.Printf("Convert embeddings... ")
	dumpWordEmbeddings(pyTorchParams["model.shared.weight"], c.model.Embeddings, c.model.Config.VocabSize)
	log.Printf("Ok\n")

	c.addToModelMapping(mapBartEncoder(c.model.Encoder))
	c.addToModelMapping(mapBartDecoder(c.model.Decoder))
	c.addToModelMapping(mapClassificationHead(c.classificationHead))
	// TODO: convert other Heads

	fmt.Printf("Setting model.encoder.embed_positions.weight.... ")
	assignToParamsList(
		pyTorchParams["model.encoder.embed_positions.weight"],
		c.model.Encoder.LearnedPositionalEmbeddings.Vectors,
		c.model.Encoder.Config.MaxPositionEmbeddings+c.model.Decoder.Config.ExtraPosEmbedding,
		c.model.Encoder.Config.DModel)
	fmt.Print("ok\n")

	fmt.Printf("Setting model.decoder.embed_positions.weight.... ")
	assignToParamsList(
		pyTorchParams["model.decoder.embed_positions.weight"],
		c.model.Decoder.LearnedPositionalEmbeddings.Vectors,
		c.model.Decoder.Config.MaxPositionEmbeddings+c.model.Decoder.Config.ExtraPosEmbedding,
		c.model.Decoder.Config.DModel)
	fmt.Print("ok\n")

	log.Printf("Search for matches with the mapped model to import weights...")
	for paramName, preTrainedWeights := range pyTorchParams {
		if param, ok := c.modelMapping[paramName]; ok {
			fmt.Printf("Setting %s...", paramName)
			if param.value.Size() != len(preTrainedWeights) {
				log.Fatal("Size mismatch")
			}
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
	fmt.Printf("BART has been successfully converted!\n")
	return nil
}

func dumpWordEmbeddings(source []float64, dest *embeddings.Model, vocabSize int) {
	size := dest.Size
	for i := 0; i < vocabSize; i++ {
		start := i * size
		end := (i + 1) * size
		data := source[start:end]
		dest.SetEmbeddingFromData(strconv.Itoa(i), data)
	}
}

func (c *huggingFacePreTrainedConverter) serializeModel() error {
	// TODO: handle different heads and base BART model alone as well
	sequenceClassificationModel := &barthead.SequenceClassification{
		BART:           c.model,
		Classification: c.classificationHead,
	}
	err := utils.SerializeToFile(c.modelFilename, nn.NewParamsSerializer(sequenceClassificationModel))
	if err != nil {
		return fmt.Errorf("bert: error during model serialization. %v", err)
	}
	fmt.Println("ok")
	return nil
}

func mapBartEncoder(model *bartencoder.Model) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	for i := 0; i < model.Config.EncoderLayers; i++ {
		layer := model.Layers.Layers[i].(*bartencoder.Layer)
		prefixBase := fmt.Sprintf("model.encoder.layers.%d", i)
		// Sublayer 1
		for j := 0; j < model.Config.EncoderAttentionHeads; j++ {
			attention := layer.SelfAttention.Attention[j]
			prefix := fmt.Sprintf("%s.%d.self_attn", prefixBase, j)
			paramsMap[fmt.Sprintf("%s.q_proj.weight", prefix)] = attention.Query.W.Value()
			paramsMap[fmt.Sprintf("%s.q_proj.bias", prefix)] = attention.Query.B.Value()
			paramsMap[fmt.Sprintf("%s.k_proj.weight", prefix)] = attention.Key.W.Value()
			paramsMap[fmt.Sprintf("%s.k_proj.bias", prefix)] = attention.Key.B.Value()
			paramsMap[fmt.Sprintf("%s.v_proj.weight", prefix)] = attention.Value.W.Value()
			paramsMap[fmt.Sprintf("%s.v_proj.bias", prefix)] = attention.Value.B.Value()
		}
		paramsMap[fmt.Sprintf("%s.self_attn.out_proj.weight", prefixBase)] = layer.SelfAttention.OutputMerge.W.Value()
		paramsMap[fmt.Sprintf("%s.self_attn.out_proj.bias", prefixBase)] = layer.SelfAttention.OutputMerge.B.Value()
		paramsMap[fmt.Sprintf("%s.self_attn_layer_norm.weight", prefixBase)] = layer.SelfAttentionLayerNorm.W.Value()
		paramsMap[fmt.Sprintf("%s.self_attn_layer_norm.bias", prefixBase)] = layer.SelfAttentionLayerNorm.B.Value()
		// Sublayer 2
		paramsMap[fmt.Sprintf("%s.fc1.weight", prefixBase)] = layer.FFN.Layers[0].(*linear.Model).W.Value()
		paramsMap[fmt.Sprintf("%s.fc1.bias", prefixBase)] = layer.FFN.Layers[0].(*linear.Model).B.Value()
		paramsMap[fmt.Sprintf("%s.fc2.weight", prefixBase)] = layer.FFN.Layers[2].(*linear.Model).W.Value()
		paramsMap[fmt.Sprintf("%s.fc2.bias", prefixBase)] = layer.FFN.Layers[2].(*linear.Model).B.Value()
		paramsMap[fmt.Sprintf("%s.final_layer_norm.weight", prefixBase)] = layer.LayerNorm.W.Value()
		paramsMap[fmt.Sprintf("%s.final_layer_norm.bias", prefixBase)] = layer.LayerNorm.B.Value()
	}

	paramsMap["model.encoder.layernorm_embedding.weight"] = model.EmbeddingLayerNorm.W.Value()
	paramsMap["model.encoder.layernorm_embedding.bias"] = model.EmbeddingLayerNorm.B.Value()
	paramsMap["model.encoder.layer_norm.weight"] = model.LayerNorm.W.Value()
	paramsMap["model.encoder.layer_norm.bias"] = model.LayerNorm.B.Value()

	return paramsMap
}

func mapBartDecoder(model *bartdecoder.Model) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	for i := 0; i < model.Config.DecoderLayers; i++ {
		layer := model.Layers.Layers[i].(*bartdecoder.Layer)
		prefixBase := fmt.Sprintf("model.decoder.layers.%d", i)
		// Self Attention
		for j := 0; j < model.Config.DecoderAttentionHeads; j++ {
			attention := layer.SelfAttention.Attention[j]
			prefix := fmt.Sprintf("%s.%d.self_attn", prefixBase, j)
			paramsMap[fmt.Sprintf("%s.q_proj.weight", prefix)] = attention.Query.W.Value()
			paramsMap[fmt.Sprintf("%s.q_proj.bias", prefix)] = attention.Query.B.Value()
			paramsMap[fmt.Sprintf("%s.k_proj.weight", prefix)] = attention.Key.W.Value()
			paramsMap[fmt.Sprintf("%s.k_proj.bias", prefix)] = attention.Key.B.Value()
			paramsMap[fmt.Sprintf("%s.v_proj.weight", prefix)] = attention.Value.W.Value()
			paramsMap[fmt.Sprintf("%s.v_proj.bias", prefix)] = attention.Value.B.Value()
		}
		paramsMap[fmt.Sprintf("%s.self_attn.out_proj.weight", prefixBase)] = layer.SelfAttention.OutputMerge.W.Value()
		paramsMap[fmt.Sprintf("%s.self_attn.out_proj.bias", prefixBase)] = layer.SelfAttention.OutputMerge.B.Value()
		paramsMap[fmt.Sprintf("%s.self_attn_layer_norm.weight", prefixBase)] = layer.SelfAttentionLayerNorm.W.Value()
		paramsMap[fmt.Sprintf("%s.self_attn_layer_norm.bias", prefixBase)] = layer.SelfAttentionLayerNorm.B.Value()

		// Cross Attention
		for j := 0; j < model.Config.DecoderAttentionHeads; j++ {
			attention := layer.EncoderAttention.Attention[j]
			prefix := fmt.Sprintf("%s.%d.encoder_attn", prefixBase, j)
			paramsMap[fmt.Sprintf("%s.q_proj.weight", prefix)] = attention.Query.W.Value()
			paramsMap[fmt.Sprintf("%s.q_proj.bias", prefix)] = attention.Query.B.Value()
			paramsMap[fmt.Sprintf("%s.k_proj.weight", prefix)] = attention.Key.W.Value()
			paramsMap[fmt.Sprintf("%s.k_proj.bias", prefix)] = attention.Key.B.Value()
			paramsMap[fmt.Sprintf("%s.v_proj.weight", prefix)] = attention.Value.W.Value()
			paramsMap[fmt.Sprintf("%s.v_proj.bias", prefix)] = attention.Value.B.Value()
		}
		paramsMap[fmt.Sprintf("%s.encoder_attn.out_proj.weight", prefixBase)] = layer.EncoderAttention.OutputMerge.W.Value()
		paramsMap[fmt.Sprintf("%s.encoder_attn.out_proj.bias", prefixBase)] = layer.EncoderAttention.OutputMerge.B.Value()
		paramsMap[fmt.Sprintf("%s.encoder_attn_layer_norm.weight", prefixBase)] = layer.EncoderAttentionLayerNorm.W.Value()
		paramsMap[fmt.Sprintf("%s.encoder_attn_layer_norm.bias", prefixBase)] = layer.EncoderAttentionLayerNorm.B.Value()

		// Sublayer 2
		paramsMap[fmt.Sprintf("%s.fc1.weight", prefixBase)] = layer.FFN.Layers[0].(*linear.Model).W.Value()
		paramsMap[fmt.Sprintf("%s.fc1.bias", prefixBase)] = layer.FFN.Layers[0].(*linear.Model).B.Value()
		paramsMap[fmt.Sprintf("%s.fc2.weight", prefixBase)] = layer.FFN.Layers[2].(*linear.Model).W.Value()
		paramsMap[fmt.Sprintf("%s.fc2.bias", prefixBase)] = layer.FFN.Layers[2].(*linear.Model).B.Value()
		paramsMap[fmt.Sprintf("%s.final_layer_norm.weight", prefixBase)] = layer.LayerNorm.W.Value()
		paramsMap[fmt.Sprintf("%s.final_layer_norm.bias", prefixBase)] = layer.LayerNorm.B.Value()
	}

	paramsMap["model.decoder.layernorm_embedding.weight"] = model.EmbeddingLayerNorm.W.Value()
	paramsMap["model.decoder.layernorm_embedding.bias"] = model.EmbeddingLayerNorm.B.Value()
	paramsMap["model.decoder.layer_norm.weight"] = model.LayerNorm.W.Value()
	paramsMap["model.decoder.layer_norm.bias"] = model.LayerNorm.B.Value()
	return paramsMap
}

func mapClassificationHead(model *barthead.Classification) map[string]mat.Matrix {
	paramsMap := make(map[string]mat.Matrix)
	paramsMap["classification_head.dense.weight"] = model.Layers[0].(*linear.Model).W.Value()
	paramsMap["classification_head.dense.bias"] = model.Layers[0].(*linear.Model).B.Value()
	paramsMap["classification_head.out_proj.weight"] = model.Layers[2].(*linear.Model).W.Value()
	paramsMap["classification_head.out_proj.bias"] = model.Layers[2].(*linear.Model).B.Value()
	return paramsMap
}

func assignToParamsList(source []float64, dest []*nn.Param, rows, cols int) {
	for i := 0; i < rows; i++ {
		dest[i].Value().SetData(source[i*cols : (i+1)*cols])
	}
}

func (c *huggingFacePreTrainedConverter) extractHuggingFaceParams() map[string][]float64 {
	paramsMap := make(map[string][]float64)
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
	c.disaggregateParams(paramsMap)
	return paramsMap
}

func (c *huggingFacePreTrainedConverter) disaggregateParams(paramsMap map[string][]float64) {
	c.disaggregateEncoderSelfAttentionParams(paramsMap)
	c.disaggregateDecoderSelfAttentionParams(paramsMap)
	c.disaggregateDecoderCrossAttentionParams(paramsMap)
}

func (c *huggingFacePreTrainedConverter) disaggregateEncoderSelfAttentionParams(paramsMap map[string][]float64) {
	for i := 0; i < c.config.EncoderLayers; i++ {
		prefix := fmt.Sprintf("model.encoder.layers.%d.self_attn", i)
		queryWeight := paramsMap[fmt.Sprintf("%s.q_proj.weight", prefix)]
		queryBias := paramsMap[fmt.Sprintf("%s.q_proj.bias", prefix)]
		keyWeight := paramsMap[fmt.Sprintf("%s.k_proj.weight", prefix)]
		keyBias := paramsMap[fmt.Sprintf("%s.k_proj.bias", prefix)]
		valueWeight := paramsMap[fmt.Sprintf("%s.v_proj.weight", prefix)]
		valueBias := paramsMap[fmt.Sprintf("%s.v_proj.bias", prefix)]
		dim := len(queryBias) / c.config.EncoderAttentionHeads
		dim2 := len(queryBias)
		for j := 0; j < c.config.EncoderAttentionHeads; j++ {
			from := j * dim
			to := (j + 1) * dim
			newPrefix := fmt.Sprintf("model.encoder.layers.%d.%d.self_attn", i, j)
			paramsMap[fmt.Sprintf("%s.q_proj.weight", newPrefix)] = queryWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.q_proj.bias", newPrefix)] = queryBias[from:to]
			paramsMap[fmt.Sprintf("%s.k_proj.weight", newPrefix)] = keyWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.k_proj.bias", newPrefix)] = keyBias[from:to]
			paramsMap[fmt.Sprintf("%s.v_proj.weight", newPrefix)] = valueWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.v_proj.bias", newPrefix)] = valueBias[from:to]
		}
	}
}

func (c *huggingFacePreTrainedConverter) disaggregateDecoderSelfAttentionParams(paramsMap map[string][]float64) {
	for i := 0; i < c.config.DecoderLayers; i++ {
		prefix := fmt.Sprintf("model.decoder.layers.%d.self_attn", i)
		queryWeight := paramsMap[fmt.Sprintf("%s.q_proj.weight", prefix)]
		queryBias := paramsMap[fmt.Sprintf("%s.q_proj.bias", prefix)]
		keyWeight := paramsMap[fmt.Sprintf("%s.k_proj.weight", prefix)]
		keyBias := paramsMap[fmt.Sprintf("%s.k_proj.bias", prefix)]
		valueWeight := paramsMap[fmt.Sprintf("%s.v_proj.weight", prefix)]
		valueBias := paramsMap[fmt.Sprintf("%s.v_proj.bias", prefix)]
		dim := len(queryBias) / c.config.DecoderAttentionHeads
		dim2 := len(queryBias)
		for j := 0; j < c.config.DecoderAttentionHeads; j++ {
			from := j * dim
			to := (j + 1) * dim
			newPrefix := fmt.Sprintf("model.decoder.layers.%d.%d.self_attn", i, j)
			paramsMap[fmt.Sprintf("%s.q_proj.weight", newPrefix)] = queryWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.q_proj.bias", newPrefix)] = queryBias[from:to]
			paramsMap[fmt.Sprintf("%s.k_proj.weight", newPrefix)] = keyWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.k_proj.bias", newPrefix)] = keyBias[from:to]
			paramsMap[fmt.Sprintf("%s.v_proj.weight", newPrefix)] = valueWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.v_proj.bias", newPrefix)] = valueBias[from:to]
		}
	}
}

func (c *huggingFacePreTrainedConverter) disaggregateDecoderCrossAttentionParams(paramsMap map[string][]float64) {
	for i := 0; i < c.config.DecoderLayers; i++ {
		prefix := fmt.Sprintf("model.decoder.layers.%d.encoder_attn", i)
		queryWeight := paramsMap[fmt.Sprintf("%s.q_proj.weight", prefix)]
		queryBias := paramsMap[fmt.Sprintf("%s.q_proj.bias", prefix)]
		keyWeight := paramsMap[fmt.Sprintf("%s.k_proj.weight", prefix)]
		keyBias := paramsMap[fmt.Sprintf("%s.k_proj.bias", prefix)]
		valueWeight := paramsMap[fmt.Sprintf("%s.v_proj.weight", prefix)]
		valueBias := paramsMap[fmt.Sprintf("%s.v_proj.bias", prefix)]
		dim := len(queryBias) / c.config.DecoderAttentionHeads
		dim2 := len(queryBias)
		for j := 0; j < c.config.DecoderAttentionHeads; j++ {
			from := j * dim
			to := (j + 1) * dim
			newPrefix := fmt.Sprintf("model.decoder.layers.%d.%d.encoder_attn", i, j)
			paramsMap[fmt.Sprintf("%s.q_proj.weight", newPrefix)] = queryWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.q_proj.bias", newPrefix)] = queryBias[from:to]
			paramsMap[fmt.Sprintf("%s.k_proj.weight", newPrefix)] = keyWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.k_proj.bias", newPrefix)] = keyBias[from:to]
			paramsMap[fmt.Sprintf("%s.v_proj.weight", newPrefix)] = valueWeight[from*dim2 : to*dim2]
			paramsMap[fmt.Sprintf("%s.v_proj.bias", newPrefix)] = valueBias[from:to]
		}
	}
}

// normalizeParamName applies the following transformation:
//    gamma -> weight
//    beta -> bias
func normalizeParamName(orig string) (normalized string) {
	normalized = orig
	normalized = strings.Replace(normalized, ".gamma", ".weight", -1)
	normalized = strings.Replace(normalized, ".beta", ".bias", -1)
	return
}

func (c *huggingFacePreTrainedConverter) addToModelMapping(paramsMap map[string]mat.Matrix) {
	for k, v := range paramsMap {
		c.modelMapping[k] = &mappedParam{
			value: v,
			used:  false,
		}
	}
}
