// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/mat"
	"strings"
)

// ForEachParam iterate all the parameters of a model also exploring the sub-parameters recursively.
func ForEachParam[T mat.DType](m Model[T], fn ParamsTraversalFunc[T]) {
	newParamsTraversal(fn, true).walk(m)
}

// ForEachParamStrict iterate all the parameters of a model without exploring the sub-models.
func ForEachParamStrict[T mat.DType](m Model[T], fn ParamsTraversalFunc[T]) {
	newParamsTraversal(fn, false).walk(m)
}

// ZeroGrad set the gradients of all model's parameters (including sub-params) to zeros.
func ZeroGrad[T mat.DType](m Model[T]) {
	ForEachParam(m, func(param Param[T], _ string, _ ParamsType) {
		param.ZeroGrad()
	})
}

// ClearSupport clears the support structure of all model's parameters (including sub-params).
func ClearSupport[T mat.DType](m Model[T]) {
	ForEachParam(m, func(param Param[T], _ string, _ ParamsType) {
		param.ClearPayload()
	})
}

// Introspect set the name property of each model's param (including sub-models).
func Introspect[T mat.DType, M Model[T]](m M) M {
	ForEachParam(Model[T](m), func(param Param[T], name string, pType ParamsType) {
		if param.Name() == "" {
			param.SetName(strings.ToLower(name))
		}
		param.SetType(pType)
	})
	return m
}

// DumpParamsVector dumps all params of a Model into a single Dense vector.
func DumpParamsVector[T mat.DType](model Model[T]) mat.Matrix[T] {
	data := make([]T, 0)
	ForEachParam(model, func(param Param[T], _ string, _ ParamsType) {
		data = append(data, param.Value().Data()...)
	})
	return mat.NewVecDense(data)
}

// LoadParamsVector sets all params of a Model from a previously dumped Dense vector.
func LoadParamsVector[T mat.DType](model Model[T], vector mat.Matrix[T]) {
	data := vector.Data()
	offset := 0
	ForEachParam(model, func(param Param[T], _ string, _ ParamsType) {
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
