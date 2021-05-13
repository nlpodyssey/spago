// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

// ProcessingMode regulates the different usage of some operations (e.g. Dropout, BatchNorm, etc.),
// depending on whether you're doing training or inference.
// Failing to set the right mode will yield inconsistent inference results.
type ProcessingMode uint8

const (
	// Training is to be used during the training phase of a model. For example, dropouts are enabled.
	Training ProcessingMode = iota
	// Inference keeps weights fixed while using the model and disables some operations (e.g. skip dropout).
	Inference
)

// Model is implemented by all neural network architectures.
// You can assign parameters (i.e. nn.Param) as regular attributes (if any).
// A Model can also contain other Model(s), allowing to nest them in a tree structure.
// Through "reification" (i.e. nn.Reify()), a Model operates as a "processor" using the computational graph.
// The Forward() operation can only be performed on a reified model (a.k.a. processor).
type Model interface {
	// Graph returns the computational graph on which the model operates (can be nil).
	Graph() *ag.Graph
	// Mode returns whether the model is being used for training or inference.
	Mode() ProcessingMode
	// IsProcessor returns whether the model has been reified (i.e., contextualized to operate
	// on a graph) and can perform the Forward().
	IsProcessor() bool
	// InitProcessor is used to initialize structures and data useful for the Forward().
	// nn.Reify() automatically invokes InitProcessor() for any sub-models.
	InitProcessor()
	// Close can be used to close or finalize model structures.
	// For example, embeddings.Model needs to close the underlying key-value store.
	Close()
}

// StandardForwarder consists of a Forward variadic function that accepts ag.Node and returns a slice of ag.Node.
// It is called StandardForwarder since this is the most frequent forward method among all implemented neural models.
type StandardForwarder interface {
	// Forward executes the forward step for each input and returns the result.
	// Recurrent networks, treats the input nodes as a sequence. Differently, feed-forward
	// networks are stateless so every computation is independent and possibly concurrent.
	Forward(xs ...ag.Node) []ag.Node
}

// StandardModel consists of a model that implements StandardForwarder.
type StandardModel interface {
	Model
	StandardForwarder
}

// Reify returns a new "reified" model (a.k.a. processor) to execute the forward step.
func Reify(m Model, g *ag.Graph, mode ProcessingMode) Model {
	return newReifier(g, mode).reify(m)
}

// ReifyForTraining returns a new reified model (a.k.a. processor) with the
// given Graph, setting the mode to Training.
func ReifyForTraining(m Model, g *ag.Graph) Model {
	return Reify(m, g, Training)
}

// ReifyForInference returns a new reified model (a.k.a. processor) with the
// given Graph, setting the mode to Inference.
func ReifyForInference(m Model, g *ag.Graph) Model {
	return Reify(m, g, Inference)
}

// ForEachParam iterate all the parameters of a model also exploring the sub-parameters recursively.
func ForEachParam(m Model, callback func(param Param)) {
	newParamsTraversal(callback, true).walk(m)
}

// ForEachParamStrict iterate all the parameters of a model without exploring the sub-models.
func ForEachParamStrict(m Model, callback func(param Param)) {
	newParamsTraversal(callback, false).walk(m)
}

// ZeroGrad set the gradients of all model's parameters (including sub-params) to zeros.
func ZeroGrad(m Model) {
	ForEachParam(m, func(param Param) {
		param.ZeroGrad()
	})
}

// ClearSupport clears the support structure of all model's parameters (including sub-params).
func ClearSupport(m Model) {
	ForEachParam(m, func(param Param) {
		param.ClearPayload()
	})
}

// DumpParamsVector dumps all params of a Model into a single Dense vector.
func DumpParamsVector(model Model) mat.Matrix {
	data := make([]mat.Float, 0)
	ForEachParam(model, func(param Param) {
		data = append(data, param.Value().Data()...)
	})
	return mat.NewVecDense(data)
}

// LoadParamsVector sets all params of a Model from a previously dumped Dense vector.
func LoadParamsVector(model Model, vector mat.Matrix) {
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
