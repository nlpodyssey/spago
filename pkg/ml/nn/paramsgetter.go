// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import "github.com/nlpodyssey/spago/pkg/mat"

// ParamsGetter is implemented by any value that has the ParamsList method,
// which should return the list of parameters of one or more models.
type ParamsGetter[T mat.DType] interface {
	Params() []Param[T]
}

var _ ParamsGetter[float32] = &DefaultParamsIterator[float32]{}

// DefaultParamsIterator is spaGO default implementation of a ParamsGetter.
type DefaultParamsIterator[T mat.DType] struct {
	models []Model[T]
}

// NewDefaultParamsIterator returns a new DefaultParamsIterator.
func NewDefaultParamsIterator[T mat.DType](models ...Model[T]) *DefaultParamsIterator[T] {
	return &DefaultParamsIterator[T]{
		models: models,
	}
}

// Params returns a slice with all Param elements from all models held by
// the DefaultParamsIterator.
func (i *DefaultParamsIterator[T]) Params() []Param[T] {
	params := make([]Param[T], 0)
	for _, model := range i.models {
		ForEachParam(model, func(param Param[T]) {
			params = append(params, param)
		})
	}
	return params
}
