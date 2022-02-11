// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/clipper"
	"github.com/nlpodyssey/spago/pkg/utils/processingqueue"
	"runtime"
	"sync"
)

// GradientDescent implements Gradients Descent (GD) optimization.
type GradientDescent[T mat.DType] struct {
	method           Method[T] // optimization method (SGD, AdaGrad, Adam, ...)
	gradClipper      clipper.GradClipper[T]
	paramsGetter     nn.ParamsGetter[T]
	paramsToOptimize []nn.Param[T]
	// processingQueue allows proper handling for computationally heavy operations
	// such as the params update step.
	// The default size is defaultProcessingQueueSize.
	processingQueue processingqueue.ProcessingQueue
}

// defaultProcessingQueueSize is the default size of GradientDescent.processingQueue on a new optimizer.
var defaultProcessingQueueSize = runtime.NumCPU()

// Option allows to configure a new GradientDescent with your specific needs.
type Option[T mat.DType] func(*GradientDescent[T])

// ClipGradByValue is an option to clip the gradients during the training between
// -value and +value.
func ClipGradByValue[T mat.DType](value T) Option[T] {
	return func(f *GradientDescent[T]) {
		f.gradClipper = &clipper.ClipValue[T]{Value: value}
	}
}

// ClipGradByNorm is an option to clip the gradients during the training by norm.
func ClipGradByNorm[T mat.DType](max, normType T) Option[T] {
	return func(f *GradientDescent[T]) {
		f.gradClipper = &clipper.ClipNorm[T]{
			MaxNorm:  max,
			NormType: normType,
		}
	}
}

// ConcurrentComputations sets the maximum number of concurrent computations handled by the GradientDescent
// for heavy tasks such as the params update steps.
// The value 1 corresponds to sequential execution.
func ConcurrentComputations[T mat.DType](value int) Option[T] {
	if value < 1 {
		panic("gd: ConcurrentComputations value must be greater than zero")
	}
	return func(f *GradientDescent[T]) {
		f.processingQueue = processingqueue.New(value)
	}
}

// NewOptimizer returns a new GradientDescent optimizer. The gradient clipper can be set to nil.
func NewOptimizer[T mat.DType](method Method[T], paramsIterator nn.ParamsGetter[T], opts ...Option[T]) *GradientDescent[T] {
	optimizer := &GradientDescent[T]{
		method:           method,
		paramsGetter:     paramsIterator,
		paramsToOptimize: make([]nn.Param[T], 0),
		processingQueue:  processingqueue.New(defaultProcessingQueueSize),
	}
	for _, opt := range opts {
		opt(optimizer)
	}
	return optimizer
}

// Optimize optimize the params, applying the optional gradient clipping.
// After the optimization the params have zero gradients.
func (o *GradientDescent[_]) Optimize() {
	o.paramsToOptimize = o.paramsGetter.Params()
	if o.paramsToOptimize == nil {
		return
	}
	o.clipGrads()
	o.updateParams()
	o.paramsToOptimize = nil
}

// updateParamsSerial applies the optimization method to all the observed parameters.
func (o *GradientDescent[_]) updateParamsSerial() {
	for _, param := range o.paramsToOptimize {
		if param.HasGrad() {
			delta := o.method.Delta(param) // important: don't release delta here
			param.ApplyDelta(delta)
			param.ZeroGrad()
		}
	}
}

// updateParams applies the optimization method to all the observed parameters concurrently.
func (o *GradientDescent[T]) updateParams() {
	var wg sync.WaitGroup
	for _, param := range o.paramsToOptimize {
		if !param.HasGrad() {
			continue
		}
		wg.Add(1)
		go func(param nn.Param[T]) {
			defer wg.Done()
			o.processingQueue.Run(func() {
				delta := o.method.Delta(param)
				param.ApplyDelta(delta)
			})
			param.ZeroGrad()
		}(param)
	}
	wg.Wait()
}

// clipGrad applies the gradient clipping to all the observed parameters.
func (o *GradientDescent[T]) clipGrads() {
	if o.gradClipper == nil {
		return
	}
	var gs []mat.Matrix[T]
	for _, param := range o.paramsToOptimize {
		if param.HasGrad() { // don't consider grad at zero
			gs = append(gs, param.Grad())
		}
	}
	o.gradClipper.Clip(gs)
}

// IncExample beats the occurrence of a new example.
func (o *GradientDescent[_]) IncExample() {
	if method, ok := o.method.(ExampleScheduler); ok {
		method.IncExample()
	}
}

// IncBatch beats the occurrence of a new batch.
func (o *GradientDescent[_]) IncBatch() {
	if method, ok := o.method.(BatchScheduler); ok {
		method.IncBatch()
	}
}

// IncEpoch beats the occurrence of a new epoch.
func (o *GradientDescent[_]) IncEpoch() {
	if method, ok := o.method.(EpochScheduler); ok {
		method.IncEpoch()
	}
}
