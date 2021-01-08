// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequencelabeler

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"github.com/nlpodyssey/gopickle/pickle"
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnncrf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/lstm"
	"github.com/nlpodyssey/spago/pkg/nlp/charlm"
	"github.com/nlpodyssey/spago/pkg/nlp/contextualstringembeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/pkg/utils/gopickleutils"
	"io"
	"io/ioutil"
	"log"
	"math"
	"path"
	"reflect"
	"sort"
)

const (
	defaultModelFilename        = "model.bin"
	defaultEmbeddingsPathPrefix = "embeddings_storage"
	defaultDictionaryFilename   = "charlm_vocab.json"
	defaultConfigFilename       = "config.json"
	defaultSequenceSeparator    = "\n"
	defaultUnknownToken         = "<unk>"
)

type converter struct {
	pathToPyTorchModel string
	tagger             *types.Dict
	params             map[string]*mappedParam
}

func newConverter(pathToPyTorchModel string) *converter {
	return &converter{
		pathToPyTorchModel: pathToPyTorchModel,
		params:             map[string]*mappedParam{},
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

// Convert converts the parameters of a Flair model into spaGO structures.
func Convert(modelPath string, flairModelName string) {
	defer embeddings.Close()

	c := newConverter(path.Join(modelPath, flairModelName))
	c.unpickleModel()

	config := c.config()
	{
		configData, err := json.MarshalIndent(config, "", "  ")
		if err != nil {
			panic(fmt.Errorf("error marshaling configuration: %v", err))
		}
		err = ioutil.WriteFile(path.Join(modelPath, defaultConfigFilename), configData, 0644)
	}

	stateDict := c.buildStateDict()

	// ---

	flairEmbeddingsForward, _ := c.embeddings().FlairEmbeddings()
	embeddingsLMForward := flairEmbeddingsForward.modules.MustGet("lm").(*flairLanguageModel)

	normalizedVocab := embeddingsLMForward.Dictionary.GetItems()
	dictData, err := json.Marshal(normalizedVocab)
	if err != nil {
		panic(fmt.Errorf("error marshaling vocab: %v", err))
	}
	err = ioutil.WriteFile(path.Join(modelPath, defaultDictionaryFilename), dictData, 0644)

	// ---

	model := NewDefaultModel(config, modelPath, false, true)

	for i, we := range c.embeddings().WordEmbeddings() {
		log.Printf("Load word embeddings %d...", i)
		c.extractWordEmbeddings(we, model.EmbeddingsLayer.WordsEncoders[i].(*embeddings.Model))
		log.Println("ok")
	}

	lmIndex := len(config.WordEmbeddings)
	lm := model.EmbeddingsLayer.WordsEncoders[lmIndex].(*contextualstringembeddings.Model).LeftToRight
	lmRev := model.EmbeddingsLayer.WordsEncoders[lmIndex].(*contextualstringembeddings.Model).RightToLeft

	if lm.Config.VocabularySize != lmRev.VocabularySize || lm.Config.EmbeddingSize != lmRev.EmbeddingSize {
		panic("language model size mismatch")
	}

	assignToParamsList(
		stateDict["lm.forward.embeddings.weight"],
		lm.Embeddings,
		lm.Config.VocabularySize,
		lm.Config.EmbeddingSize)

	assignToParamsList(
		stateDict["lm.backward.embeddings.weight"],
		lmRev.Embeddings,
		lmRev.Config.VocabularySize,
		lmRev.Config.EmbeddingSize)

	c.mapCharLM(lm, "lm.forward.")
	c.mapCharLM(lmRev, "lm.backward.")
	c.mapLinear(model.EmbeddingsLayer.ProjectionLayer, "embeddings_projection.")
	c.mapTagger(model.TaggerLayer)

	// ---

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

	output := path.Join(modelPath, config.ModelFilename)
	log.Printf("Serializing full model to \"%s\"... ", output)
	err = utils.SerializeToFile(output, nn.NewParamsSerializer(model))
	if err != nil {
		panic("error during model serialization.")
	}
	log.Println("ok")

	log.Println("Conversion done!")
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

func extractLSTMParams(lm *flairLanguageModel, keyPrefix string, dest map[string][]mat.Float) {
	params := lm.modules.MustGet("rnn").(*torchLSTM).parameters

	wII, wIF, wIC, wIO := chunkTensorBy4(wrapTensor(params.MustGet("weight_ih_l0")))
	wHI, wHF, wHC, wHO := chunkTensorBy4(wrapTensor(params.MustGet("weight_hh_l0")))
	bHI, bHF, bHC, bHO := chunkTensorBy4(wrapTensor(params.MustGet("bias_hh_l0")))
	bII, bIF, bIC, bIO := chunkTensorBy4(wrapTensor(params.MustGet("bias_ih_l0")))

	dest[keyPrefix+"lstm.w_ii"] = wII
	dest[keyPrefix+"lstm.w_hi"] = wHI
	dest[keyPrefix+"lstm.b_i"] = sumVectors(bHI, bII)
	dest[keyPrefix+"lstm.w_if"] = wIF
	dest[keyPrefix+"lstm.w_hf"] = wHF
	dest[keyPrefix+"lstm.b_f"] = sumVectors(bHF, bIF)
	dest[keyPrefix+"lstm.w_io"] = wIO
	dest[keyPrefix+"lstm.w_ho"] = wHO
	dest[keyPrefix+"lstm.b_o"] = sumVectors(bHO, bIO)
	dest[keyPrefix+"lstm.w_ic"] = wIC
	dest[keyPrefix+"lstm.w_hc"] = wHC
	dest[keyPrefix+"lstm.b_c"] = sumVectors(bHC, bIC)
}

func (c *converter) embeddings() *stackedEmbeddings {
	return c.tagger.MustGet("embeddings").(*stackedEmbeddings)
}

func (c *converter) buildStateDict() map[string][]mat.Float {
	flairEmbeddingsForward, flairEmbeddingsBackward := c.embeddings().FlairEmbeddings()

	embeddingsLMForward := flairEmbeddingsForward.modules.MustGet("lm").(*flairLanguageModel)
	forwardDecoderParams := embeddingsLMForward.modules.MustGet("decoder").(*torchLinear).parameters
	forwardEncoderParams := embeddingsLMForward.modules.MustGet("encoder").(*sparseEmbedding).parameters

	embeddingsLMBackward := flairEmbeddingsBackward.modules.MustGet("lm").(*flairLanguageModel)
	backwardDecoderParams := embeddingsLMBackward.modules.MustGet("decoder").(*torchLinear).parameters
	backwardEncoderParams := embeddingsLMBackward.modules.MustGet("encoder").(*sparseEmbedding).parameters

	taggerStateDict := c.tagger.MustGet("state_dict").(*types.OrderedDict)

	if !reflect.DeepEqual(embeddingsLMForward.Dictionary.IdxToItem, embeddingsLMBackward.Dictionary.IdxToItem) {
		panic(fmt.Errorf("different charlm and charlm_rev dictionaries"))
	}

	stateDict := make(map[string][]mat.Float)

	stateDict["lm.forward.decoder.weight"] = gopickleutils.GetData(wrapTensor(forwardDecoderParams.MustGet("weight")))
	stateDict["lm.forward.decoder.bias"] = gopickleutils.GetData(wrapTensor(forwardDecoderParams.MustGet("bias")))
	stateDict["lm.forward.embeddings.weight"] = gopickleutils.GetData(wrapTensor(forwardEncoderParams.MustGet("weight")))
	extractLSTMParams(embeddingsLMForward, "lm.forward.", stateDict)

	if proj, ok := embeddingsLMForward.modules.Get("proj"); ok && proj != nil {
		forwardLMProjParams := proj.(*torchLinear).parameters
		stateDict["lm.forward.projection.weight"] = gopickleutils.GetData(wrapTensor(forwardLMProjParams.MustGet("weight")))
		stateDict["lm.forward.projection.bias"] = gopickleutils.GetData(wrapTensor(forwardLMProjParams.MustGet("bias")))
	}

	stateDict["lm.backward.decoder.weight"] = gopickleutils.GetData(wrapTensor(backwardDecoderParams.MustGet("weight")))
	stateDict["lm.backward.decoder.bias"] = gopickleutils.GetData(wrapTensor(backwardDecoderParams.MustGet("bias")))
	stateDict["lm.backward.embeddings.weight"] = gopickleutils.GetData(wrapTensor(backwardEncoderParams.MustGet("weight")))
	extractLSTMParams(embeddingsLMBackward, "lm.backward.", stateDict)

	if proj, ok := embeddingsLMBackward.modules.Get("proj"); ok && proj != nil {
		backwardLMProjParams := proj.(*torchLinear).parameters
		stateDict["lm.backward.projection.weight"] = gopickleutils.GetData(wrapTensor(backwardLMProjParams.MustGet("weight")))
		stateDict["lm.backward.projection.bias"] = gopickleutils.GetData(wrapTensor(backwardLMProjParams.MustGet("bias")))
	}

	stateDict["embeddings_projection.weight"] = gopickleutils.GetData(taggerStateDict.MustGet("embedding2nn.weight").(*pytorch.Tensor))
	stateDict["embeddings_projection.bias"] = gopickleutils.GetData(taggerStateDict.MustGet("embedding2nn.bias").(*pytorch.Tensor))

	c.extractBiLSTMParams(stateDict)

	stateDict["scorer.weight"] = gopickleutils.GetData(taggerStateDict.MustGet("linear.weight").(*pytorch.Tensor))
	stateDict["scorer.bias"] = gopickleutils.GetData(taggerStateDict.MustGet("linear.bias").(*pytorch.Tensor))

	stateDict["crf.transitions"] = c.transitionWeights()
	return stateDict
}

func (c *converter) extractBiLSTMParams(dest map[string][]mat.Float) {
	StateDict := c.tagger.MustGet("state_dict").(*types.OrderedDict)

	wII, wIF, wIC, wIO := chunkTensorBy4(StateDict.MustGet("rnn.weight_ih_l0").(*pytorch.Tensor))
	wHI, wHF, wHC, wHO := chunkTensorBy4(StateDict.MustGet("rnn.weight_hh_l0").(*pytorch.Tensor))
	bHI, bHF, bHC, bHO := chunkTensorBy4(StateDict.MustGet("rnn.bias_hh_l0").(*pytorch.Tensor))
	bII, bIF, bIC, bIO := chunkTensorBy4(StateDict.MustGet("rnn.bias_ih_l0").(*pytorch.Tensor))

	revWII, revWIF, revWIC, revWIO := chunkTensorBy4(StateDict.MustGet("rnn.weight_ih_l0_reverse").(*pytorch.Tensor))
	revWHI, revWHF, revWHC, revWHO := chunkTensorBy4(StateDict.MustGet("rnn.weight_hh_l0_reverse").(*pytorch.Tensor))
	revBHI, revBHF, revBHC, revBHO := chunkTensorBy4(StateDict.MustGet("rnn.bias_hh_l0_reverse").(*pytorch.Tensor))
	revBII, revBIF, revBIC, revBIO := chunkTensorBy4(StateDict.MustGet("rnn.bias_ih_l0_reverse").(*pytorch.Tensor))

	dest["bilstm.forward.w_ii"] = wII
	dest["bilstm.forward.w_hi"] = wHI
	dest["bilstm.forward.b_i"] = sumVectors(bHI, bII)
	dest["bilstm.forward.w_if"] = wIF
	dest["bilstm.forward.w_hf"] = wHF
	dest["bilstm.forward.b_f"] = sumVectors(bHF, bIF)
	dest["bilstm.forward.w_io"] = wIO
	dest["bilstm.forward.w_ho"] = wHO
	dest["bilstm.forward.b_o"] = sumVectors(bHO, bIO)
	dest["bilstm.forward.w_ic"] = wIC
	dest["bilstm.forward.w_hc"] = wHC
	dest["bilstm.forward.b_c"] = sumVectors(bHC, bIC)

	dest["bilstm.backward.w_ii"] = revWII
	dest["bilstm.backward.w_hi"] = revWHI
	dest["bilstm.backward.b_i"] = sumVectors(revBHI, revBII)
	dest["bilstm.backward.w_if"] = revWIF
	dest["bilstm.backward.w_hf"] = revWHF
	dest["bilstm.backward.b_f"] = sumVectors(revBHF, revBIF)
	dest["bilstm.backward.w_io"] = revWIO
	dest["bilstm.backward.w_ho"] = revWHO
	dest["bilstm.backward.b_o"] = sumVectors(revBHO, revBIO)
	dest["bilstm.backward.w_ic"] = revWIC
	dest["bilstm.backward.w_hc"] = revWHC
	dest["bilstm.backward.b_c"] = sumVectors(revBHC, revBIC)
}

func (c *converter) transitionWeights() []mat.Float {
	labels := c.tagger.MustGet("tag_dictionary").(*flairDictionary).GetItems()
	labelsSize := len(labels)
	weights := gopickleutils.GetData(c.tagger.MustGet("state_dict").(*types.OrderedDict).MustGet("transitions").(*pytorch.Tensor))

	// TODO: is it guaranteed that "<START>" and "<STOP>" are always the last two items?
	startIndex := len(labels) - 2
	stopIndex := len(labels) - 1

	length := len(labels) - 1
	out := make([]mat.Float, length*length)
	for i := range out {
		out[i] = -10000
	}

	i := 0

	for rowIndex := 0; rowIndex < labelsSize; rowIndex++ {
		row := weights[rowIndex*labelsSize : rowIndex*labelsSize+labelsSize]
		j := 0
		if i != startIndex { // skip transition ending in start
			if i < startIndex {
				for _, col := range row {
					if j != stopIndex { // skip transition starting in end
						if j == startIndex { // transition starting at start
							out[i+1] = col
						} else {
							out[(j+1)*length+(i+1)] = col
						}
					}
					j++
				}
			} else {
				for _, col := range row {
					if j != stopIndex { // skip transition starting in end
						if j == startIndex { // transition starting at start
							out[0] = col
						} else {
							out[(j+1)*length] = col
						}
					}
					j++
				}
			}
		}
		i++
	}

	return out
}

func (c *converter) extractWordEmbeddings(we *wordEmbeddings, dest *embeddings.Model) {
	vectorSize := we.PrecomputedWordEmbeddings.Vectors.Shape[1]
	data := we.PrecomputedWordEmbeddings.Vectors.FloatSlice()

	for index, item := range *we.PrecomputedWordEmbeddings.IndexToWord {
		word := item.(string)
		if len(word) == 0 {
			continue // skip empty words
		}

		vector := data[index*vectorSize : index*vectorSize+vectorSize]
		dest.SetEmbeddingFromData(word, vector)
	}
}

var unpicklerClasses = map[string]interface{}{
	"flair.embeddings.StackedEmbeddings":              stackedEmbeddingsClass{},
	"flair.embeddings.WordEmbeddings":                 wordEmbeddingsClass{},
	"flair.embeddings.FlairEmbeddings":                flairEmbeddingsClass{},
	"flair.models.language_model.LanguageModel":       flairLanguageModelClass{},
	"gensim.models.keyedvectors.Word2VecKeyedVectors": word2VecKeyedVectorsClass{},
	"gensim.models.keyedvectors.Vocab":                vocabClass{},
	"numpy.core.multiarray._reconstruct":              multiarrayReconstruct{},
	"numpy.ndarray":                                   ndarrayClass{},
	"numpy.dtype":                                     dtypeClass{},
	"torch.nn.modules.dropout.Dropout":                dropoutClass{},
	"torch.nn.modules.sparse.Embedding":               sparseEmbeddingClass{},
	"torch._utils._rebuild_parameter":                 rebuildParameter{},
	"torch.nn.modules.rnn.LSTM":                       torchLSTMClass{},
	"torch.nn.modules.linear.Linear":                  torchLinearClass{},
	"flair.data.Dictionary":                           flairDictionaryClass{},
	"torch.backends.cudnn.rnn.Unserializable":         torchRNNUnserializableClass{},
}

func unpicklerFindClass(module, name string) (interface{}, error) {
	c, ok := unpicklerClasses[fmt.Sprintf("%s.%s", module, name)]
	if !ok {
		return nil, fmt.Errorf("class not found: %s %s", module, name)
	}
	return c, nil
}

func (c *converter) unpickleModel() {
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

func (c *converter) config() Config {
	// we don't care if it's forward or backward, we just need the dimensions
	fembs, _ := c.embeddings().FlairEmbeddings()
	embeddingsLM := fembs.modules.MustGet("lm").(*flairLanguageModel)
	embeddingsLMRNN := embeddingsLM.modules.MustGet("rnn").(*torchLSTM)

	stateDict := c.tagger.MustGet("state_dict").(*types.OrderedDict)
	embedding2NNWeight := stateDict.MustGet("embedding2nn.weight").(*pytorch.Tensor)
	rnnWeightIH := stateDict.MustGet("rnn.weight_ih_l0").(*pytorch.Tensor)
	rnnWeightHH := stateDict.MustGet("rnn.weight_hh_l0").(*pytorch.Tensor)
	linearWeight := stateDict.MustGet("linear.weight").(*pytorch.Tensor)

	labels := c.tagger.MustGet("tag_dictionary").(*flairDictionary).GetItems()
	// Exclude "<START>" and "<STOP>"
	// TODO: is it guaranteed that "<START>" and "<STOP>" are always the last two items?
	labels = labels[:len(labels)-2]

	wembs := c.embeddings().WordEmbeddings()
	wordEmbeddingsConfigs := make([]WordEmbeddingsConfig, len(wembs))
	for i, we := range wembs {
		wordEmbeddingsConfigs[i] = WordEmbeddingsConfig{
			WordEmbeddingsFilename: fmt.Sprintf("%s_%d", defaultEmbeddingsPathPrefix, i),
			WordEmbeddingsSize:     we.PrecomputedWordEmbeddings.VectorSize,
		}
	}

	return Config{
		ModelFilename:  defaultModelFilename,
		WordEmbeddings: wordEmbeddingsConfigs,
		ContextualStringEmbeddings: ContextualEmbeddingsConfig{
			VocabularySize:     embeddingsLM.Dictionary.Len(),
			EmbeddingSize:      embeddingsLMRNN.InputSize,
			HiddenSize:         embeddingsLMRNN.HiddenSize,
			OutputSize:         embeddingsLM.NOut,
			SequenceSeparator:  defaultSequenceSeparator, // TODO: use "document_delimiter" from embeddingsLM, if it exists
			UnknownToken:       defaultUnknownToken,      // TODO: it should be the first entry from EmbeddingsLM.Dictionary
			VocabularyFilename: defaultDictionaryFilename,
		},
		EmbeddingsProjectionInputSize:  embedding2NNWeight.Size[1],
		EmbeddingsProjectionOutputSize: embedding2NNWeight.Size[0],
		RecurrentInputSize:             rnnWeightIH.Size[1],
		RecurrentOutputSize:            rnnWeightHH.Size[1],
		ScorerInputSize:                linearWeight.Size[1],
		ScorerOutputSize:               linearWeight.Size[0],
		Labels:                         labels,
	}
}

func assignToParamsList(source []mat.Float, dest []nn.Param, rows, cols int) {
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
	c.mapLSTM(model.RNN, fmt.Sprintf("%slstm.", prefix))
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

func wrapTensor(value interface{}) *pytorch.Tensor {
	switch vv := value.(type) {
	case *pytorch.Tensor:
		return vv
	case *parameter:
		return vv.Data
	default:
		panic(fmt.Errorf("cannot wrap *pytorch.Tensor with type %T: %#v", value, value))
	}
}

// Stacked Embeddings

type stackedEmbeddingsClass struct{}

var _ types.PyNewable = stackedEmbeddingsClass{}
var _ types.PyDictSettable = &stackedEmbeddings{}

type stackedEmbeddings struct {
	Embeddings *types.List
	pyDict     map[string]interface{}
}

func (c stackedEmbeddingsClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("stackedEmbeddingsClass: unsupported arguments: %#v", args)
	}
	return &stackedEmbeddings{
		pyDict: make(map[string]interface{}),
	}, nil
}

func (s *stackedEmbeddings) PyDictSet(key, value interface{}) error {
	k := key.(string)
	switch k {
	case "embeddings":
		s.Embeddings = value.(*types.List)
	default:
		s.pyDict[k] = value
	}
	return nil
}

type wordEmbeddingsByName []*wordEmbeddings

func (e wordEmbeddingsByName) Len() int           { return len(e) }
func (e wordEmbeddingsByName) Swap(i, j int)      { e[i], e[j] = e[j], e[i] }
func (e wordEmbeddingsByName) Less(i, j int) bool { return e[i].name < e[j].name }

func (s *stackedEmbeddings) WordEmbeddings() []*wordEmbeddings {
	result := make([]*wordEmbeddings, 0, s.Embeddings.Len())
	for _, e := range *s.Embeddings {
		if we, ok := e.(*wordEmbeddings); ok {
			result = append(result, we)
		}
	}

	// Word embeddings must be sorted by name, alphabetically
	sort.Sort(wordEmbeddingsByName(result))

	return result
}

func (s *stackedEmbeddings) FlairEmbeddings() (forward *flairEmbeddings, backward *flairEmbeddings) {
	for _, e := range *s.Embeddings {
		fe, isFlairEmbeddings := e.(*flairEmbeddings)
		if !isFlairEmbeddings {
			continue
		}
		if fe.isForwardLM {
			if forward != nil {
				panic("stackedEmbeddings: unexpected multiple forward flairEmbeddings")
			}
			forward = fe
		} else {
			if backward != nil {
				panic("stackedEmbeddings: unexpected multiple backward flairEmbeddings")
			}
			backward = fe
		}
	}
	if forward == nil {
		panic("stackedEmbeddings: missing forward flairEmbeddings")
	}
	if backward == nil {
		panic("stackedEmbeddingsmissing  backward flairEmbeddings")
	}
	return forward, backward
}

// Word Embeddings

type wordEmbeddingsClass struct{}
type wordEmbeddings struct {
	PrecomputedWordEmbeddings *word2VecKeyedVectors
	name                      string
	pyDict                    map[string]interface{}
}

var _ types.PyNewable = wordEmbeddingsClass{}
var _ types.PyDictSettable = &wordEmbeddings{}

func (c wordEmbeddingsClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("wordEmbeddingsClass: unsupported arguments: %#v", args)
	}
	return &wordEmbeddings{
		pyDict: make(map[string]interface{}),
	}, nil
}

func (w *wordEmbeddings) PyDictSet(key, value interface{}) error {
	k := key.(string)
	switch k {
	case "precomputed_word_embeddings":
		w.PrecomputedWordEmbeddings = value.(*word2VecKeyedVectors)
	case "name":
		w.name = value.(string)
	default:
		w.pyDict[k] = value
	}
	return nil
}

// Flair Embeddings

type flairEmbeddingsClass struct{}
type flairEmbeddings struct {
	name        string
	isForwardLM bool
	modules     *types.OrderedDict

	pyDict map[string]interface{}
}

var _ types.PyNewable = flairEmbeddingsClass{}
var _ types.PyDictSettable = &flairEmbeddings{}

func (c flairEmbeddingsClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("flairEmbeddingsClass: unsupported arguments: %#v", args)
	}
	return &flairEmbeddings{
		pyDict: make(map[string]interface{}),
	}, nil
}

func (f *flairEmbeddings) PyDictSet(key, value interface{}) error {
	k := key.(string)
	switch k {
	case "name":
		f.name = value.(string)
	case "is_forward_lm":
		f.isForwardLM = value.(bool)
	case "_modules":
		f.modules = value.(*types.OrderedDict)
	default:
		f.pyDict[k] = value
	}
	return nil
}

// Flair Language Model

type flairLanguageModelClass struct{}
type flairLanguageModel struct {
	Dictionary *flairDictionary
	NOut       int
	modules    *types.OrderedDict
	pyDict     map[string]interface{}
}

var _ types.PyNewable = flairLanguageModelClass{}
var _ types.PyDictSettable = &flairLanguageModel{}

func (c flairLanguageModelClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("flairLanguageModelClass: unsupported arguments: %#v", args)
	}
	return &flairLanguageModel{
		NOut:   -1,
		pyDict: make(map[string]interface{}),
	}, nil
}

func (f *flairLanguageModel) PyDictSet(key, value interface{}) error {
	k := key.(string)
	switch k {
	case "_modules":
		f.modules = value.(*types.OrderedDict)
	case "dictionary":
		f.Dictionary = value.(*flairDictionary)
	case "nout":
		switch v := value.(type) {
		case nil:
			f.NOut = -1
		case int:
			f.NOut = v
		default:
			return fmt.Errorf("unsupported nout value: %#v", value)
		}
	default:
		f.pyDict[k] = value
	}
	return nil
}

// PyTorch Dropout

type dropoutClass struct{}
type dropout struct {
	pyDict map[string]interface{}
}

var _ types.PyNewable = dropoutClass{}
var _ types.PyDictSettable = &dropout{}

func (c dropoutClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("dropoutClass: unsupported arguments: %#v", args)
	}
	return &dropout{
		pyDict: make(map[string]interface{}),
	}, nil
}

func (w *dropout) PyDictSet(key, value interface{}) error {
	w.pyDict[key.(string)] = value
	return nil
}

// PyTorch LSTM

type torchLSTMClass struct{}
type torchLSTM struct {
	InputSize  int
	HiddenSize int
	parameters *types.OrderedDict
	pyDict     map[string]interface{}
}

var _ types.PyNewable = torchLSTMClass{}
var _ types.PyDictSettable = &torchLSTM{}

func (c torchLSTMClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("torchLSTMClass: unsupported arguments: %#v", args)
	}
	return &torchLSTM{
		pyDict: make(map[string]interface{}),
	}, nil
}

func (t *torchLSTM) PyDictSet(key, value interface{}) error {
	k := key.(string)
	switch k {
	case "_parameters":
		t.parameters = value.(*types.OrderedDict)
	case "input_size":
		t.InputSize = value.(int)
	case "hidden_size":
		t.HiddenSize = value.(int)
	default:
		t.pyDict[k] = value
	}
	return nil
}

// PyTorch Linear

type torchLinearClass struct{}
type torchLinear struct {
	parameters *types.OrderedDict
	pyDict     map[string]interface{}
}

var _ types.PyNewable = torchLinearClass{}
var _ types.PyDictSettable = &torchLinear{}

func (c torchLinearClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("torchLinearClass: unsupported arguments: %#v", args)
	}
	return &torchLinear{
		pyDict: make(map[string]interface{}),
	}, nil
}

func (t *torchLinear) PyDictSet(key, value interface{}) error {
	k := key.(string)
	switch k {
	case "_parameters":
		t.parameters = value.(*types.OrderedDict)
	default:
		t.pyDict[k] = value
	}
	return nil
}

// PyTorch Linear

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

// PyTorch Sparse Embedding

type sparseEmbeddingClass struct{}
type sparseEmbedding struct {
	parameters *types.OrderedDict
	pyDict     map[string]interface{}
}

var _ types.PyNewable = sparseEmbeddingClass{}
var _ types.PyDictSettable = &sparseEmbedding{}

func (c sparseEmbeddingClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("sparseEmbeddingClass: unsupported arguments: %#v", args)
	}
	return &sparseEmbedding{
		pyDict: make(map[string]interface{}),
	}, nil
}

func (s *sparseEmbedding) PyDictSet(key, value interface{}) error {
	k := key.(string)
	switch k {
	case "_parameters":
		s.parameters = value.(*types.OrderedDict)
	default:
		s.pyDict[k] = value
	}
	return nil
}

// Word2VecKeyedVectors

type word2VecKeyedVectorsClass struct{}
type word2VecKeyedVectors struct {
	Vectors     *ndarray
	IndexToWord *types.List
	VectorSize  int
	pyDict      map[string]interface{}
}

var _ types.PyNewable = word2VecKeyedVectorsClass{}
var _ types.PyDictSettable = &word2VecKeyedVectors{}

func (c word2VecKeyedVectorsClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("word2VecKeyedVectorsClass: unsupported arguments: %#v", args)
	}
	return &word2VecKeyedVectors{
		pyDict: make(map[string]interface{}),
	}, nil
}

func (w *word2VecKeyedVectors) PyDictSet(key, value interface{}) error {
	k := key.(string)
	switch k {
	case "vectors":
		w.Vectors = value.(*ndarray)
	case "index2word":
		w.IndexToWord = value.(*types.List)
	case "vector_size":
		w.VectorSize = value.(int)
	default:
		w.pyDict[k] = value
	}
	return nil
}

// Word2VecKeyedVectors

type vocabClass struct{}
type vocab struct {
	pyDict map[string]interface{}
}

var _ types.PyNewable = vocabClass{}
var _ types.PyDictSettable = &vocab{}

func (c vocabClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("vocabClass: unsupported arguments: %#v", args)
	}
	return &vocab{
		pyDict: make(map[string]interface{}),
	}, nil
}

func (v *vocab) PyDictSet(key, value interface{}) error {
	v.pyDict[key.(string)] = value
	return nil
}

// Numpy Multiarray Reconstruct

type multiarrayReconstruct struct{}

var _ types.Callable = multiarrayReconstruct{}

// Call construct an empty array.
func (multiarrayReconstruct) Call(args ...interface{}) (interface{}, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("multiarrayReconstruct: three arguments expected, actual: %#v", args)
	}
	subType := args[0].(types.PyNewable)
	shape := args[1].(*types.Tuple)
	dataType := string(args[2].([]uint8))
	return subType.PyNew(shape, dataType)
}

// Torch utils _rebuildParameter

type parameter struct {
	Data          *pytorch.Tensor
	RequiresGrad  bool
	backwardHooks *types.OrderedDict
}

type rebuildParameter struct{}

var _ types.Callable = rebuildParameter{}

// Call construct an empty array.
func (rebuildParameter) Call(args ...interface{}) (interface{}, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("rebuildParameter: three arguments expected, actual: %#v", args)
	}
	return &parameter{
		Data:         args[0].(*pytorch.Tensor),
		RequiresGrad: args[1].(bool),
		// NB: This exists only for backwards compatibility; the
		// general expectation is that backward_hooks is an empty
		// OrderedDict.  See Note [Don't serialize hooks]
		backwardHooks: args[2].(*types.OrderedDict),
	}, nil
}

// Numpy ndarray

type ndarrayClass struct{}
type ndarray struct {
	Shape    []int
	DataType *dtypeInstance
	rawData  []uint8
}

var _ types.PyNewable = ndarrayClass{}
var _ types.PyStateSettable = &ndarray{}

func (c ndarrayClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) == 0 || len(args) > 2 {
		return nil, fmt.Errorf("ndarrayClass: expected one or two arguments, actual: %#v", args)
	}

	shapeTuple := args[0].(*types.Tuple)
	shape := make([]int, shapeTuple.Len())
	for i := range shape {
		shape[i] = shapeTuple.Get(i).(int)
	}

	dataType := ""
	if len(args) > 1 {
		dataType = args[1].(string)
	}

	return &ndarray{
		Shape: shape,
		DataType: &dtypeInstance{
			Value: dataType,
		},
		rawData: nil,
	}, nil
}

func (n *ndarray) PySetState(state interface{}) error {
	t := state.(*types.Tuple)

	// [0] version : int - optional pickle version. If omitted defaults to 0.
	// [1] shape : tuple
	shapeTuple := t.Get(1).(*types.Tuple)
	n.Shape = make([]int, shapeTuple.Len())
	for i := range n.Shape {
		n.Shape[i] = shapeTuple.Get(i).(int)
	}
	// [2] dtype : data-type
	n.DataType = t.Get(2).(*dtypeInstance)
	// [3] isFortran : bool
	// [4] rawdata : string or list - a binary string with the data (or a list if 'a' is an object array)
	n.rawData = t.Get(4).([]uint8)
	return nil
}

func (n *ndarray) FloatSlice() []mat.Float {
	dataType := n.DataType.Value.(string)
	if dataType != "f4" {
		panic(fmt.Errorf("ndarray.FloatSlice(): only DataType `f4` is supported, actual: %v", dataType))
	}
	span := 4
	length := len(n.rawData)

	if length%span != 0 {
		panic(fmt.Errorf("ndarray.FloatSlice(): cannot use span %d on raw data with length %d", span, len(n.rawData)))
	}

	result := make([]mat.Float, 0, length/span)
	for i := 0; i < length; i += span {
		bytes := n.rawData[i : i+span]
		result = append(result, mat.Float(math.Float32frombits(binary.LittleEndian.Uint32(bytes))))
	}
	return result
}

// Numpy dtype

type dtypeClass struct{}
type dtypeInstance struct {
	Value interface{}
	Align bool
	Copy  bool
	state interface{}
}

var _ types.PyNewable = dtypeClass{}
var _ types.Callable = dtypeClass{}
var _ types.PyStateSettable = &dtypeInstance{}

func (c dtypeClass) Call(args ...interface{}) (interface{}, error) {
	return c.PyNew(args...)
}

func (c dtypeClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) == 0 || len(args) > 3 {
		return nil, fmt.Errorf("dtypeClass: expected one to three arguments, actual: %#v", args)
	}

	dt := &dtypeInstance{
		Value: args[0],
		Align: false,
		Copy:  false,
		state: nil,
	}

	if len(args) > 1 {
		switch t := args[1].(type) {
		case int:
			dt.Align = t == 1
		case bool:
			dt.Align = t
		}
	}

	if len(args) > 2 {
		switch t := args[2].(type) {
		case int:
			dt.Copy = t == 1
		case bool:
			dt.Copy = t
		}
	}

	return dt, nil
}

func (d *dtypeInstance) PySetState(state interface{}) error {
	d.state = state
	return nil
}

// pytorch rnn Unserializable

type torchRNNUnserializableClass struct{}
type torchRNNUnserializable struct{}

var _ types.PyNewable = torchRNNUnserializableClass{}

func (c torchRNNUnserializableClass) PyNew(args ...interface{}) (interface{}, error) {
	return &torchRNNUnserializable{}, nil
}
