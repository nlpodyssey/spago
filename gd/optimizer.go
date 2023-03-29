// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

import (
	"runtime"

	"github.com/nlpodyssey/spago/gd/clipper"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

// Optimizer implements Gradients Descent (GD) optimization.
type Optimizer struct {
	model       nn.Model // model to optimize
	method      Method   // optimization method (SGD, AdaGrad, Adam, ...)
	gradClipper clipper.GradClipper
}

// NewOptimizer returns a new Optimizer.
func NewOptimizer(model nn.Model, method Method) *Optimizer {
	optimizer := &Optimizer{
		model:  model,
		method: method,
	}
	return optimizer
}

// WithClipGradByValue is an option to clip the gradients during the training between
// -value and +value.
func (o *Optimizer) WithClipGradByValue(value float64) *Optimizer {
	o.gradClipper = &clipper.ClipValue{Value: value}
	return o
}

// WithClipGradByNorm is an option to clip the gradients during the training by norm.
func (o *Optimizer) WithClipGradByNorm(max, normType float64) *Optimizer {
	o.gradClipper = &clipper.ClipNorm{
		MaxNorm:  max,
		NormType: normType,
	}
	return o
}

// Optimize optimizes the model parameters, applying the optional gradient clipping.
// After the optimization the params have zero gradients.
func (o *Optimizer) Optimize() {
	params := o.collectParams()
	o.clipGradsInPlace(params)
	o.updateParams(params)
}

func (o *Optimizer) collectParams() []nn.Param {
	visited := map[nn.Param]struct{}{}
	params := make([]nn.Param, 0)
	nn.ForEachParam(o.model, func(param nn.Param) {
		if !param.HasGrad() {
			return // don't consider params with grad at zero
		}
		if _, ok := visited[param]; !ok {
			params = append(params, param)
			visited[param] = struct{}{}
		}
	})
	return params
}

// updateParams applies the optimization method to all the observed parameters.
func (o *Optimizer) updateParams(params []nn.Param) {
	ch := make(chan struct{}, runtime.NumCPU())
	for _, param := range params {
		ch <- struct{}{}
		go func(p nn.Param) {
			delta := o.method.Delta(p)
			p.ApplyDelta(delta)
			p.ZeroGrad()
			<-ch
		}(param)
	}
	for i := 0; i < len(ch); i++ {
		ch <- struct{}{}
	}
	close(ch)
}

// clipGrad applies the gradient clipping to all the observed parameters.
func (o *Optimizer) clipGradsInPlace(params []nn.Param) {
	if o.gradClipper == nil {
		return
	}
	var gs []mat.Matrix
	for _, param := range params {
		gs = append(gs, param.Grad())
	}
	o.gradClipper.Clip(gs)
}

// IncExample beats the occurrence of a new example.
func (o *Optimizer) IncExample() {
	if method, ok := o.method.(interface{ IncExample() }); ok {
		method.IncExample()
	}
}

// IncBatch beats the occurrence of a new batch.
func (o *Optimizer) IncBatch() {
	if method, ok := o.method.(interface{ IncBatch() }); ok {
		method.IncBatch()
	}
}

// IncEpoch beats the occurrence of a new epoch.
func (o *Optimizer) IncEpoch() {
	if method, ok := o.method.(interface{ IncEpoch() }); ok {
		method.IncEpoch()
	}
}
