// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimizers

import (
	"context"
	"runtime"
	"sync"

	"github.com/nlpodyssey/spago/nn"
)

// OptimizationStrategy is the interface implemented by AdaGrad, Adam, etc.
type OptimizationStrategy interface {
	OptimizeParams(*nn.Param) error
}

// Optimizer is an optimizer that can optimize a set of parameters.
type Optimizer struct {
	// parameters is a function that returns a channel of parameters to optimize.
	parameters nn.ParamChannelFunc
	// strategy is the optimization strategy to use.
	strategy OptimizationStrategy
}

// New returns a new optimizer.
func New(parameters nn.ParamChannelFunc, strategy OptimizationStrategy) *Optimizer {
	return &Optimizer{
		parameters: parameters,
		strategy:   strategy,
	}
}

// Optimize performs the optimization of the parameters.
func (o *Optimizer) Optimize() error {
	var wg sync.WaitGroup
	guard := make(chan struct{}, runtime.NumCPU()*2)
	errCh := make(chan error, 1)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for param := range o.parameters(ctx) {
		select {
		case err := <-errCh:
			cancel()  // As soon as an error occurs, stop the iteration over parameters
			wg.Wait() // Wait for running goroutines to finish
			return err
		default:
			param := param
			wg.Add(1)
			guard <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-guard }()
				if !param.HasGrad() {
					return
				}
				if err := o.strategy.OptimizeParams(param); err != nil {
					select {
					case errCh <- err:
					default:
					}
				}
			}()
		}
	}

	close(errCh)

	if err, ok := <-errCh; ok {
		return err
	}

	return nil
}
