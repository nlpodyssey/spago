// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// Model is implemented by all neural network architectures.
// You can assign parameters (i.e. nn.Param) as regular attributes (if any).
// A Model can also contain other Model(s), allowing to nest them in a tree structure.
// Through "reification" (i.e. nn.Reify()), a Model operates as a "processor" using the computational graph.
// The Forward() operation can only be performed on a reified model (a.k.a. processor).
type Model[T mat.DType] interface {
	// Graph returns the computational graph on which the model operates (can be nil).
	Graph() *ag.Graph[T]
	// InitProcessor is used to initialize structures and data useful for the Forward().
	// nn.Reify() automatically invokes InitProcessor() for any sub-models.
	InitProcessor()
}

// StandardModel consists of a model that implements a Forward variadic function that accepts ag.Node and returns a slice of ag.Node.
// It is called StandardModel since this is the most frequent forward method among all implemented neural models.
type StandardModel[T mat.DType] interface {
	Model[T]

	// Forward executes the forward step for each input and returns the result.
	// Recurrent networks, treats the input nodes as a sequence. Differently, feed-forward
	// networks are stateless so every computation is independent and possibly concurrent.
	Forward(xs ...ag.Node[T]) []ag.Node[T]
}

// Reify returns a new "reified" model (a.k.a. processor) to execute the forward step.
func Reify[T mat.DType, M Model[T]](m M, g *ag.Graph[T]) M {
	return newReifier(g).reify(m).(M)
}

// ForEachParam iterate all the parameters of a model also exploring the sub-parameters recursively.
func ForEachParam[T mat.DType](m Model[T], callback func(param Param[T])) {
	newParamsTraversal(callback, true).walk(m)
}

// ForEachParamStrict iterate all the parameters of a model without exploring the sub-models.
func ForEachParamStrict[T mat.DType](m Model[T], callback func(param Param[T])) {
	newParamsTraversal(callback, false).walk(m)
}

// ZeroGrad set the gradients of all model's parameters (including sub-params) to zeros.
func ZeroGrad[T mat.DType](m Model[T]) {
	ForEachParam(m, func(param Param[T]) {
		param.ZeroGrad()
	})
}

// ClearSupport clears the support structure of all model's parameters (including sub-params).
func ClearSupport[T mat.DType](m Model[T]) {
	ForEachParam(m, func(param Param[T]) {
		param.ClearPayload()
	})
}

// DumpParamsVector dumps all params of a Model into a single Dense vector.
func DumpParamsVector[T mat.DType](model Model[T]) mat.Matrix[T] {
	data := make([]T, 0)
	ForEachParam(model, func(param Param[T]) {
		data = append(data, param.Value().Data()...)
	})
	return mat.NewVecDense(data)
}

// LoadParamsVector sets all params of a Model from a previously dumped Dense vector.
func LoadParamsVector[T mat.DType](model Model[T], vector mat.Matrix[T]) {
	data := vector.Data()
	offset := 0
	ForEachParam(model, func(param Param[T]) {
		size := param.Value().Size()
		param.Value().SetData(data[offset : offset+size])
		offset += size
	})
}

// MakeNewModels return n new models.
// The callback is delegated to return a new model for each i-item.
func MakeNewModels[T mat.DType](n int, callback func(i int) Model[T]) []Model[T] {
	lst := make([]Model[T], n)
	for i := 0; i < n; i++ {
		lst[i] = callback(i)
	}
	return lst
}
