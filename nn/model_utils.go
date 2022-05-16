// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

// ForEachParam iterate all the parameters of a model also exploring the sub-parameters recursively.
func ForEachParam(m Model, fn ParamsTraversalFunc) {
	newParamsTraversal(fn, true).walk(m)
}

// ForEachParamStrict iterate all the parameters of a model without exploring the sub-models.
func ForEachParamStrict(m Model, fn ParamsTraversalFunc) {
	newParamsTraversal(fn, false).walk(m)
}

// ZeroGrad set the gradients of all model's parameters (including sub-params) to zeros.
func ZeroGrad(m Model) {
	ForEachParam(m, func(param Param, _ string, _ ParamsType) {
		param.ZeroGrad()
	})
}

// ClearSupport clears the support structure of all model's parameters (including sub-params).
func ClearSupport(m Model) {
	ForEachParam(m, func(param Param, _ string, _ ParamsType) {
		param.ClearPayload()
	})
}

// Init set the name property of each model's param (including sub-models).
func Init[M Model](m M) M {
	ForEachParam(Model(m), func(param Param, name string, pType ParamsType) {
		if p, ok := param.(ParamNameSetter); ok && param.Name() == "" {
			p.SetName(name)
		}
		if p, ok := param.(ParamTypeSetter); ok {
			p.SetType(pType)
		}
	})
	return m
}
