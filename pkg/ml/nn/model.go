// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/utils"
	"io"
)

// ProcessingMode regulates the different usage of some operations (e.g. Dropout, BatchNorm, etc.) inside a Processor,
// depending on whether you're doing training or inference.
// Failing to set the right mode will yield inconsistent inference results.
type ProcessingMode int

const (
	// Training is to be used during the training phase of a model. For example, dropouts are enabled.
	Training ProcessingMode = iota
	// Inference keeps weights fixed while using the model and disables some operations (e.g. skip dropout).
	Inference
)

// Context is used to instantiate a processor to operate on a graph, according to the desired ProcessingMode.
// If a processor contains other sub-processors, you must instantiate them using the same context to make sure
// you are operating on the same graph and in the same mode.
type Context struct {
	// Graph is the computational graph on which the processor(s) operate.
	Graph *ag.Graph
	// Mode regulates the different usage of some operations whether you're doing training or inference.
	Mode ProcessingMode
}

// Module is the main interface defining a neural module, combining the Model and Processor interfaces.
type Module interface {
	Model
	Processor
}

// Model contains the serializable parameters.
type Model interface{}

// Processor performs the operations on the computational graphs using the model's parameters.
type Processor interface {
	// GetGraph returns the computational graph on which the processor operates (can be nil).
	GetGraph() *ag.Graph
	// GetMode returns whether the processor is being used for training or inference.
	GetMode() ProcessingMode
	// RequiresFullSeq returns whether the processor needs the complete sequence to start processing
	// (as in the case of BiRNN and other bidirectional models), or not.
	RequiresFullSeq() bool
	// Forward performs the operations on the computational graphs using the model's parameters.
	// It executes the forward step for each input and returns the result.
	// Recurrent networks treats the input nodes as a sequence.
	// Differently, feed-forward networks are stateless so every computation is independent.
	Forward(xs ...ag.Node) []ag.Node
	//
	InitProc()
}

// NewProc returns a new contextualized model to execute the forward step.
func NewProc(ctx Context, m Model) Processor {
	return newModelContextualizer(ctx).contextualize(m)
}

// ForEachParam iterate all the parameters of a model also exploring the sub-parameters recursively.
// TODO: don't loop the field every time, use a lazy initialized "params list" instead (?)
func ForEachParam(m Model, callback func(param Param)) {
	newParamsTraversal(callback, true).walk(m)
}

// ForEachParamStrict iterate all the parameters of a model without exploring the sub-models.
func ForEachParamStrict(m Model, callback func(param Param)) {
	newParamsTraversal(callback, false).walk(m)
}

// ParamsIterator is implemented by any value that has the ParamsList method,
// which should return the list of parameters of one or more models.
type ParamsIterator interface {
	ParamsList() []Param
}

var _ ParamsIterator = &DefaultParamsIterator{}

// DefaultParamsIterator is spaGO default implementation of a ParamsIterator.
type DefaultParamsIterator struct {
	models []Model
}

// NewDefaultParamsIterator returns a new DefaultParamsIterator.
func NewDefaultParamsIterator(models ...Model) *DefaultParamsIterator {
	return &DefaultParamsIterator{models: models}
}

// ParamsList returns a slice with all Param elements from all models held by
// the DefaultParamsIterator.
func (i *DefaultParamsIterator) ParamsList() []Param {
	params := make([]Param, 0)
	for _, model := range i.models {
		ForEachParam(model, func(param Param) {
			params = append(params, param)
		})
	}
	return params
}

// ZeroGrad set the gradients of all model's parameters (including sub-params) to zeros.
// TODO: use ParamsIterator?
func ZeroGrad(m Model) {
	ForEachParam(m, func(param Param) {
		param.ZeroGrad()
	})
}

// ClearSupport clears the support structure of all model's parameters (including sub-params).
// TODO: use ParamsIterator?
func ClearSupport(m Model) {
	ForEachParam(m, func(param Param) {
		param.ClearPayload()
	})
}

// DumpParamsVector dumps all params of a Model into a single Dense vector.
// TODO: use ParamsIterator?
func DumpParamsVector(model Model) *mat.Dense {
	data := make([]float64, 0)
	ForEachParam(model, func(param Param) {
		data = append(data, param.Value().Data()...)
	})
	return mat.NewVecDense(data)
}

// LoadParamsVector sets all params of a Model from a previously dumped Dense vector.
// TODO: use ParamsIterator?
func LoadParamsVector(model Model, vector *mat.Dense) {
	data := vector.Data()
	offset := 0
	ForEachParam(model, func(param Param) {
		size := param.Value().Size()
		param.Value().SetData(data[offset : offset+size])
		offset += size
	})
}

// MakeNewModels return n new models.
// The callback is delegated to return a new model for each i-item.
func MakeNewModels(n int, callback func(i int) Model) []Model {
	lst := make([]Model, n)
	for i := 0; i < n; i++ {
		lst[i] = callback(i)
	}
	return lst
}

var _ utils.Serializer = &ParamsSerializer{}
var _ utils.Deserializer = &ParamsSerializer{}

// ParamsSerializer allows serialization and deserialization of all
// parameters of a given Model.
type ParamsSerializer struct {
	Model
}

// NewParamsSerializer returns a new ParamsSerializer.
func NewParamsSerializer(m Model) *ParamsSerializer {
	return &ParamsSerializer{Model: m}
}

// Serialize dumps the params values to the writer.
// TODO: use ParamsIterator?
func (m *ParamsSerializer) Serialize(w io.Writer) (n int, err error) {
	ForEachParam(m, func(param Param) {
		cnt, err2 := mat.MarshalBinaryTo(param.Value(), w)
		n += cnt
		if err2 != nil {
			err = err2
			return
		}
	})
	return n, err
}

// Deserialize assigns the params with the values obtained from the reader.
// TODO: use ParamsIterator?
func (m *ParamsSerializer) Deserialize(r io.Reader) (n int, err error) {
	ForEachParam(m, func(param Param) {
		cnt, err2 := mat.UnmarshalBinaryFrom(param.Value(), r)
		n += cnt
		if err2 != nil {
			err = err2
			return
		}
	})
	return n, err
}

// BaseModel satisfies some methods of the Model interface.
// It is meant to be embedded in other processors to reduce the amount of boilerplate code.
type BaseModel struct {
	Ctx               Context
	FullSeqProcessing bool
}

// NewBaseModel returns a processor containing a new instance of the so-called "contextualized" model,
// in which the parameters are wrapped as graph nodes.
func NewBaseModel(ctx Context, fullSeqProcessing bool) BaseModel {
	return BaseModel{
		Ctx:               ctx,
		FullSeqProcessing: fullSeqProcessing,
	}
}

// GetMode returns whether the processor is being used for training or inference.
func (m *BaseModel) GetMode() ProcessingMode {
	return m.Ctx.Mode
}

// GetGraph returns the computational graph on which the processor operates.
func (m *BaseModel) GetGraph() *ag.Graph {
	return m.Ctx.Graph
}

// RequiresFullSeq returns whether the model needs the complete sequence to start processing
// (as in the case of BiRNN and other bidirectional models), or not.
func (m *BaseModel) RequiresFullSeq() bool {
	return m.FullSeqProcessing
}

func (m *BaseModel) InitProc() {}
