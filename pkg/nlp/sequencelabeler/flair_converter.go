// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequencelabeler

import (
	"fmt"
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnncrf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/rec/lstm"
	"github.com/nlpodyssey/spago/pkg/nlp/charlm"
	"github.com/nlpodyssey/spago/pkg/nlp/contextualstringembeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/pkg/utils/gopickleutils"
	"log"
	"path"
	"path/filepath"
)

// TODO: This code needs to be refactored. Pull requests are welcome!

const (
	defaultConfigFilename           = "config.json"
	defaultEmbeddingsFilename       = "embeddings.txt"
	defaultSecondEmbeddingsFilename = "embeddings2.txt"
)

type converter struct {
	pathToPyTorchModel string
	pyTorchParams      map[string][]float64
	params             map[string]*mappedParam
	vocabulary         []string
}

func newConverter(pathToPyTorchModel string) *converter {
	return &converter{
		pathToPyTorchModel: pathToPyTorchModel,
		pyTorchParams:      make(map[string][]float64),
		params:             map[string]*mappedParam{},
		vocabulary:         make([]string, 0),
	}
}

type mappedParam struct {
	value mat.Matrix
	used  bool
}

func newParam(value mat.Matrix) *mappedParam {
	return &mappedParam{
		value: value,
		used:  false,
	}
}

// Convert converts the parameters (weights and bias) of a pre-processed Flair model into spaGO structures.
// At this moment it is not possible to import directly from the Flair model: a simple Python script takes care
// of pre-processing and exporting the tensors in a format more compatible with spaGO. I'll make that script
// available soon, now it's a bit chaotic. In the future it would be even better to import directly from Flair.
func Convert(modelPath string, flairModelName string) {
	config := LoadConfig(path.Join(modelPath, defaultConfigFilename))
	model := NewDefaultModel(config, modelPath, false, true)
	defer embeddings.Close()

	c := newConverter(path.Join(modelPath, flairModelName))
	c.loadPyTorchParams()

	lmIndex := 0

	if config.WordEmbeddings.WordEmbeddingsSize > 0 {
		log.Printf("Load word embeddings...")
		model.EmbeddingsLayer.WordsEncoders[0].(*embeddings.Model).Load(filepath.Join(modelPath, defaultEmbeddingsFilename))
		println("ok")
		lmIndex++
	}

	if config.WordEmbeddings2.WordEmbeddingsSize > 0 {
		log.Printf("Load word embeddings...")
		model.EmbeddingsLayer.WordsEncoders[1].(*embeddings.Model).Load(filepath.Join(modelPath, defaultSecondEmbeddingsFilename))
		println("ok")
		lmIndex++
	}

	lm := model.EmbeddingsLayer.WordsEncoders[lmIndex].(*contextualstringembeddings.Model).LeftToRight
	lmRev := model.EmbeddingsLayer.WordsEncoders[lmIndex].(*contextualstringembeddings.Model).RightToLeft

	if lm.Config.VocabularySize != lmRev.VocabularySize || lm.Config.EmbeddingSize != lmRev.EmbeddingSize {
		panic("language model size mismatch.")
	}

	assignToParamsList(
		c.pyTorchParams["lm.forward.embeddings.weight"],
		lm.Embeddings,
		lm.Config.VocabularySize,
		lm.Config.EmbeddingSize)

	assignToParamsList(
		c.pyTorchParams["lm.backward.embeddings.weight"],
		lmRev.Embeddings,
		lmRev.Config.VocabularySize,
		lmRev.Config.EmbeddingSize)

	c.mapCharLM(lm, "lm.forward.")
	c.mapCharLM(lmRev, "lm.backward.")
	c.mapLinear(model.EmbeddingsLayer.ProjectionLayer, "embeddings_projection.")
	c.mapTagger(model.TaggerLayer)

	/////////////////////

	log.Printf("Search for matches with the mapped model to import weights...")
	for paramName, preTrainedWeights := range c.pyTorchParams {
		if param, ok := c.params[paramName]; ok {
			fmt.Printf("Setting %s...", paramName)
			param.value.SetData(preTrainedWeights)
			param.used = true
			fmt.Println("ok")
		} else {
			log.Printf("WARNING!! %s not found", paramName)
		}
	}

	for key, value := range c.params {
		if !value.used {
			log.Printf("WARNING!! %s not used", key)
		}
	}

	output := path.Join(modelPath, config.ModelFilename)
	fmt.Printf("Serializing full model to \"%s\"... ", output)
	err := utils.SerializeToFile(output, nn.NewParamsSerializer(model))
	if err != nil {
		panic("error during model serialization.")
	}
	fmt.Println("ok")
}

func (c *converter) loadPyTorchParams() {
	result, err := pytorch.Load(c.pathToPyTorchModel)
	if err != nil {
		log.Fatal(err)
	}
	od := result.(*types.OrderedDict)
	for key, entry := range od.Map {
		switch value := entry.Value.(type) {
		case *pytorch.Tensor:
			paramName := key.(string)
			fmt.Printf("Reading %s.... ", paramName)
			c.pyTorchParams[paramName] = gopickleutils.GetData(value)
			fmt.Println("ok")
		case *types.List:
			if entry.Key.(string) != "vocab" {
				panic("unknown entry")
			}
			length := value.Len()
			for i := 0; i < length; i++ {
				c.vocabulary = append(c.vocabulary, string(value.Get(i).([]byte)))
			}
		}
	}
}

func assignToParamsList(source []float64, dest []*nn.Param, rows, cols int) {
	for i := 0; i < rows; i++ {
		dest[i].Value().SetData(source[i*cols : (i+1)*cols])
	}
}

func (c *converter) mapTagger(model *birnncrf.Model) {
	c.mapBiLSTM(model.BiRNN, "")
	c.mapLinear(model.Scorer, "scorer.")
	c.params["crf.transitions"] = newParam(model.CRF.TransitionScores.Value())
}

func (c *converter) mapCharLM(model *charlm.Model, prefix string) {
	c.mapLSTM(model.RNN.(*lstm.Model), fmt.Sprintf("%slstm.", prefix))
	c.mapLinear(model.Decoder, fmt.Sprintf("%sdecoder.", prefix))
	if model.Config.OutputSize > 0 {
		c.mapLinear(model.Projection, fmt.Sprintf("%sprojection.", prefix))
	}
}

func (c *converter) mapLSTM(model *lstm.Model, prefix string) {
	c.params[fmt.Sprintf("%sw_ii", prefix)] = newParam(model.WIn.Value())
	c.params[fmt.Sprintf("%sw_hi", prefix)] = newParam(model.WInRec.Value())
	c.params[fmt.Sprintf("%sb_i", prefix)] = newParam(model.BIn.Value())
	c.params[fmt.Sprintf("%sw_if", prefix)] = newParam(model.WFor.Value())
	c.params[fmt.Sprintf("%sw_hf", prefix)] = newParam(model.WForRec.Value())
	c.params[fmt.Sprintf("%sb_f", prefix)] = newParam(model.BFor.Value())
	c.params[fmt.Sprintf("%sw_io", prefix)] = newParam(model.WOut.Value())
	c.params[fmt.Sprintf("%sw_ho", prefix)] = newParam(model.WOutRec.Value())
	c.params[fmt.Sprintf("%sb_o", prefix)] = newParam(model.BOut.Value())
	c.params[fmt.Sprintf("%sw_ic", prefix)] = newParam(model.WCand.Value())
	c.params[fmt.Sprintf("%sw_hc", prefix)] = newParam(model.WCandRec.Value())
	c.params[fmt.Sprintf("%sb_c", prefix)] = newParam(model.BCand.Value())
}

func (c *converter) mapBiLSTM(model *birnn.Model, prefix string) {
	c.mapLSTM(model.Positive.(*lstm.Model), fmt.Sprintf("%sbilstm.forward.", prefix))
	c.mapLSTM(model.Negative.(*lstm.Model), fmt.Sprintf("%sbilstm.backward.", prefix))
}

func (c *converter) mapLinear(model *linear.Model, prefix string) {
	c.params[fmt.Sprintf("%sweight", prefix)] = newParam(model.W.Value())
	c.params[fmt.Sprintf("%sbias", prefix)] = newParam(model.B.Value())
}
