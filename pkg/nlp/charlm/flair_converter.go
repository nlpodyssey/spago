// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	"encoding/json"
	"fmt"
	"github.com/nlpodyssey/gopickle/pickle"
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/lstm"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/pkg/utils/gopickleutils"
	"io"
	"log"
	"os"
	"path"
)

const (
	defaultModelFilename     = "model.bin"
	defaultConfigFilename    = "config.json"
	defaultSequenceSeparator = "\n"
	defaultUnknownToken      = "<unk>"
)

type flairConverter struct {
	pathToPyTorchModel string
	tagger             *types.Dict
	params             map[string]*mappedParam
}

func newConverter(pathToPyTorchModel string) *flairConverter {
	return &flairConverter{
		pathToPyTorchModel: pathToPyTorchModel,
		params:             map[string]*mappedParam{},
	}
}

type mappedParam struct {
	value mat.Matrix[mat.Float]
	used  bool
}

func newParam(value mat.Matrix[mat.Float]) *mappedParam {
	return &mappedParam{
		value: value,
		used:  false,
	}
}

// Convert converts the parameters of a Flair model into spaGO structures.
func Convert(modelPath string, flairModelName, configFileName, modelFileName string) {
	defer embeddings.Close()

	if configFileName == "" {
		configFileName = defaultConfigFilename
	}
	if modelFileName == "" {
		modelFileName = defaultModelFilename
	}

	c := newConverter(path.Join(modelPath, flairModelName))
	c.unpickleModel()

	config := c.buildConfig()
	{
		configData, err := json.MarshalIndent(config, "", "  ")
		if err != nil {
			panic(fmt.Errorf("error marshaling configuration: %w", err))
		}
		err = os.WriteFile(path.Join(modelPath, configFileName), configData, 0644)
	}

	stateDict := c.buildStateDict()

	lm := New(config)

	dict := c.tagger.MustGet("dictionary").(*flairDictionary)
	lm.Vocabulary = vocabulary.New(dict.GetItems())

	assignToParamsList(stateDict["embeddings.weight"],
		lm.Embeddings,
		lm.Config.VocabularySize,
		lm.Config.EmbeddingSize)
	delete(stateDict, "embeddings.weight") // already assigned

	c.mapLSTM(lm.RNN, fmt.Sprintf("%slstm.", ""))
	c.mapLinear(lm.Decoder, fmt.Sprintf("%sdecoder.", ""))

	log.Printf("Search for matches with the mapped model to import weights...")
	for paramName, preTrainedWeights := range stateDict {
		if param, ok := c.params[paramName]; ok {
			log.Printf("Setting %s...", paramName)
			param.value.SetData(preTrainedWeights)
			param.used = true
			log.Println("ok")
		} else {
			log.Printf("WARNING!! %s not found", paramName)
		}
	}

	for key, value := range c.params {
		if !value.used {
			log.Printf("WARNING!! %s not used", key)
		}
	}

	output := path.Join(modelPath, modelFileName)
	log.Printf("Serializing full model to \"%s\"... ", output)
	err := utils.SerializeToFile(output, lm)
	if err != nil {
		panic("error during model serialization.")
	}
	log.Println("ok")

	log.Println("Conversion done!")
}

func (c *flairConverter) buildConfig() Config {
	dictionary := c.tagger.MustGet("dictionary").(*flairDictionary)
	embeddingSize := c.tagger.MustGet("embedding_size").(int)
	hiddenSize := c.tagger.MustGet("hidden_size").(int)

	return Config{
		VocabularySize:    dictionary.Len(),
		EmbeddingSize:     embeddingSize,
		HiddenSize:        hiddenSize,
		OutputSize:        0, // TODO: check if nout is always nil
		SequenceSeparator: defaultSequenceSeparator,
		UnknownToken:      defaultUnknownToken,
	}
}

func (c *flairConverter) buildStateDict() map[string][]mat.Float {
	stateDict := make(map[string][]mat.Float)
	lmStateDict := c.tagger.MustGet("state_dict").(*types.OrderedDict)
	stateDict["embeddings.weight"] = gopickleutils.GetData(lmStateDict.MustGet("encoder.weight").(*pytorch.Tensor))
	stateDict["decoder.weight"] = gopickleutils.GetData(lmStateDict.MustGet("decoder.weight").(*pytorch.Tensor))
	stateDict["decoder.bias"] = gopickleutils.GetData(lmStateDict.MustGet("decoder.bias").(*pytorch.Tensor))
	extractLSTMParams(lmStateDict, stateDict)
	return stateDict
}

func (c *flairConverter) mapLSTM(model *lstm.Model, prefix string) {
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

func (c *flairConverter) mapLinear(model *linear.Model, prefix string) {
	c.params[fmt.Sprintf("%sweight", prefix)] = newParam(model.W.Value())
	c.params[fmt.Sprintf("%sbias", prefix)] = newParam(model.B.Value())
}

func extractLSTMParams(orig *types.OrderedDict, dest map[string][]mat.Float) {
	wII, wIF, wIC, wIO := chunkTensorBy4(orig.MustGet("rnn.weight_ih_l0").(*pytorch.Tensor))
	wHI, wHF, wHC, wHO := chunkTensorBy4(orig.MustGet("rnn.weight_hh_l0").(*pytorch.Tensor))
	bHI, bHF, bHC, bHO := chunkTensorBy4(orig.MustGet("rnn.bias_hh_l0").(*pytorch.Tensor))
	bII, bIF, bIC, bIO := chunkTensorBy4(orig.MustGet("rnn.bias_ih_l0").(*pytorch.Tensor))

	dest["lstm.w_ii"] = wII
	dest["lstm.w_hi"] = wHI
	dest["lstm.b_i"] = sumVectors(bHI, bII)
	dest["lstm.w_if"] = wIF
	dest["lstm.w_hf"] = wHF
	dest["lstm.b_f"] = sumVectors(bHF, bIF)
	dest["lstm.w_io"] = wIO
	dest["lstm.w_ho"] = wHO
	dest["lstm.b_o"] = sumVectors(bHO, bIO)
	dest["lstm.w_ic"] = wIC
	dest["lstm.w_hc"] = wHC
	dest["lstm.b_c"] = sumVectors(bHC, bIC)
}

func chunkTensor(t *pytorch.Tensor, chunks int) [][]mat.Float {
	data := gopickleutils.GetData(t)

	size := len(data)
	if size%chunks != 0 {
		panic(fmt.Errorf("cannot chunk tenson of size %+v (%d) in %d parts", t.Size, size, chunks))
	}

	chunkSize := size / chunks
	result := make([][]mat.Float, chunks)
	for i := range result {
		result[i] = data[i*chunkSize : i*chunkSize+chunkSize]
	}
	return result
}

func chunkTensorBy4(t *pytorch.Tensor) ([]mat.Float, []mat.Float, []mat.Float, []mat.Float) {
	c := chunkTensor(t, 4)
	return c[0], c[1], c[2], c[3]
}

func sumVectors(a, b []mat.Float) []mat.Float {
	if len(a) != len(b) {
		panic(fmt.Errorf("cannot sum vectors with different size: %d, %d", len(a), len(b)))
	}
	y := make([]mat.Float, len(a))
	for i, av := range a {
		y[i] = av + b[i]
	}
	return y
}

func assignToParamsList(source []mat.Float, dest []nn.Param, rows, cols int) {
	for i := 0; i < rows; i++ {
		dest[i].Value().SetData(source[i*cols : (i+1)*cols])
	}
}

func (c *flairConverter) unpickleModel() {
	newUnpickler := func(r io.Reader) pickle.Unpickler {
		u := pickle.NewUnpickler(r)
		u.FindClass = unpicklerFindClass
		return u
	}

	result, err := pytorch.LoadWithUnpickler(c.pathToPyTorchModel, newUnpickler)
	if err != nil {
		log.Fatal(err)
	}
	c.tagger = result.(*types.Dict)
}

var unpicklerClasses = map[string]interface{}{
	"flair.data.Dictionary": flairDictionaryClass{},
}

func unpicklerFindClass(module, name string) (interface{}, error) {
	c, ok := unpicklerClasses[fmt.Sprintf("%s.%s", module, name)]
	if !ok {
		return nil, fmt.Errorf("class not found: %s %s", module, name)
	}
	return c, nil
}

type flairDictionaryClass struct{}
type flairDictionary struct {
	IdxToItem *types.List
	pyDict    map[string]interface{}
}

var _ types.PyNewable = flairDictionaryClass{}
var _ types.PyDictSettable = &flairDictionary{}

func (c flairDictionaryClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("flairDictionaryClass: unsupported arguments: %#v", args)
	}
	return &flairDictionary{
		pyDict: make(map[string]interface{}),
	}, nil
}

func (f *flairDictionary) PyDictSet(key, value interface{}) error {
	k := key.(string)
	switch k {
	case "idx2item":
		f.IdxToItem = value.(*types.List)
	default:
		f.pyDict[k] = value
	}
	return nil
}

func (f *flairDictionary) Len() int {
	return f.IdxToItem.Len()
}

func (f *flairDictionary) GetItems() []string {
	items := make([]string, f.IdxToItem.Len())
	for i := range items {
		items[i] = string(f.IdxToItem.Get(i).([]uint8))
	}
	return items
}
