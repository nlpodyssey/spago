// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/mat"
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
		if p, ok := param.(ParamNameSetter); ok && param.Name() == "" {
			p.SetName(name)
		}
		if p, ok := param.(ParamTypeSetter); ok {
			p.SetType(pType)
		}
	})
	return m
}
