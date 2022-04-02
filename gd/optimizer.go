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
type Optimizer[T mat.DType] struct {
	model       nn.Model[T] // model to optimize
	method      Method[T]   // optimization method (SGD, AdaGrad, Adam, ...)
	gradClipper clipper.GradClipper[T]
}

// Option allows to configure a new Optimizer with your specific needs.
type Option[T mat.DType] func(*Optimizer[T])

// WithClipGradByValue is an option to clip the gradients during the training between
// -value and +value.
func WithClipGradByValue[T mat.DType](value T) Option[T] {
	return func(f *Optimizer[T]) {
		f.gradClipper = &clipper.ClipValue[T]{Value: value}
	}
}

// WithClipGradByNorm is an option to clip the gradients during the training by norm.
func WithClipGradByNorm[T mat.DType](max, normType T) Option[T] {
	return func(f *Optimizer[T]) {
		f.gradClipper = &clipper.ClipNorm[T]{
			MaxNorm:  max,
			NormType: normType,
		}
	}
}

// NewOptimizer returns a new Optimizer.
func NewOptimizer[T mat.DType](model nn.Model[T], method Method[T], opts ...Option[T]) *Optimizer[T] {
	optimizer := &Optimizer[T]{
		model:  model,
		method: method,
	}
	for _, opt := range opts {
		opt(optimizer)
	}
	return optimizer
}

// Optimize optimizes the model parameters, applying the optional gradient clipping.
// After the optimization the params have zero gradients.
func (o *Optimizer[T]) Optimize() {
	params := o.collectParams()
	o.clipGradsInPlace(params)
	o.updateParams(params)
}

func (o *Optimizer[T]) collectParams() []nn.Param[T] {
	visited := map[nn.Param[T]]struct{}{}
	params := make([]nn.Param[T], 0)
	nn.ForEachParam(o.model, func(param nn.Param[T], _ string, _ nn.ParamsType) {
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
func (o *Optimizer[T]) updateParams(params []nn.Param[T]) {
	ch := make(chan struct{}, runtime.NumCPU())
	for _, param := range params {
		ch <- struct{}{}
		go func(p nn.Param[T]) {
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
func (o *Optimizer[T]) clipGradsInPlace(params []nn.Param[T]) {
	if o.gradClipper == nil {
		return
	}
	var gs []mat.Matrix[T]
	for _, param := range params {
		gs = append(gs, param.Grad())
	}
	o.gradClipper.Clip(gs)
}

// IncExample beats the occurrence of a new example.
func (o *Optimizer[_]) IncExample() {
	if method, ok := o.method.(ExampleScheduler); ok {
		method.IncExample()
	}
}

// IncBatch beats the occurrence of a new batch.
func (o *Optimizer[_]) IncBatch() {
	if method, ok := o.method.(BatchScheduler); ok {
		method.IncBatch()
	}
}

// IncEpoch beats the occurrence of a new epoch.
func (o *Optimizer[_]) IncEpoch() {
	if method, ok := o.method.(EpochScheduler); ok {
		method.IncEpoch()
	}
}
