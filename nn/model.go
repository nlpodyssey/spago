// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"context"

	"github.com/nlpodyssey/spago/mat"
)

// Model is implemented by all neural network architectures.
type Model interface {
	mustEmbedModule()
}

// Apply fn recursively to every sub-models as well as self.
// Typical use includes initializing the parameters of a model.
func Apply(m Model, fn func(model Model)) {
	fn(m)
	paramsTraversal{
		paramsFunc:       nil,
		modelsFunc:       fn,
		exploreSubModels: true,
	}.walk(m)
}

type ParamChannelFunc func(ctx context.Context) <-chan *Param

func Parameters(m Model) ParamChannelFunc {
	return func(ctx context.Context) <-chan *Param {
		paramChan := make(chan *Param)

		go func() {
			defer close(paramChan)
			paramsTraversal{
				paramsFunc: func(param *Param) {
					select {
					case <-ctx.Done():
						return // Stop sending to the channel if context is done
					case paramChan <- param:
					}
				},
				modelsFunc:       nil,
				exploreSubModels: true,
			}.walk(m)
		}()

		return paramChan
	}
}

func StreamParams(params []*Param) ParamChannelFunc {
	return func(ctx context.Context) <-chan *Param {
		paramChan := make(chan *Param)

		go func() {
			defer close(paramChan)

			for _, param := range params {
				select {
				case <-ctx.Done():
					return // Stop if context is done
				case paramChan <- param:
				}
			}
		}()

		return paramChan
	}
}

// ForEachParam iterate all the parameters of a model also exploring the sub-parameters recursively.
func ForEachParam(m Model, fn func(param *Param)) {
	paramsTraversal{
		paramsFunc:       fn,
		modelsFunc:       nil,
		exploreSubModels: true,
	}.walk(m)
}

// ForEachParamStrict iterate all the parameters of a model without exploring the sub-models.
func ForEachParamStrict(m Model, fn func(param *Param)) {
	paramsTraversal{
		paramsFunc:       fn,
		modelsFunc:       nil,
		exploreSubModels: false,
	}.walk(m)
}

// ZeroGrad set the gradients of all model's parameters (including sub-params) to zeros.
func ZeroGrad(m Model) {
	ForEachParam(m, func(param *Param) {
		param.ZeroGrad()
	})
}

// StandardModel consists of a model that implements a Forward variadic function that accepts mat.Tensor and returns a slice of mat.Tensor.
// It is called StandardModel since this is the most frequent forward method among all implemented neural models.
type StandardModel interface {
	Model

	// Forward executes the forward step of the model.
	Forward(...mat.Tensor) []mat.Tensor
}

type ModuleList[T StandardModel] []T

// Forward operates on a slice of StandardModel connecting outputs to inputs sequentially for each module following,
// finally returning its output.
func (ml ModuleList[T]) Forward(xs ...mat.Tensor) []mat.Tensor {
	for _, m := range ml {
		xs = m.Forward(xs...)
	}
	return xs
}
