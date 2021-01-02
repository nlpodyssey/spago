// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

// ParamsGetter is implemented by any value that has the ParamsList method,
// which should return the list of parameters of one or more models.
type ParamsGetter interface {
	Params() []Param
}

var _ ParamsGetter = &DefaultParamsIterator{}

// DefaultParamsIterator is spaGO default implementation of a ParamsGetter.
type DefaultParamsIterator struct {
	models []Model
}

// NewDefaultParamsIterator returns a new DefaultParamsIterator.
func NewDefaultParamsIterator(models ...Model) *DefaultParamsIterator {
	return &DefaultParamsIterator{
		models: models,
	}
}

// Params returns a slice with all Param elements from all models held by
// the DefaultParamsIterator.
func (i *DefaultParamsIterator) Params() []Param {
	params := make([]Param, 0)
	for _, model := range i.models {
		ForEachParam(model, func(param Param) {
			params = append(params, param)
		})
	}
	return params
}
