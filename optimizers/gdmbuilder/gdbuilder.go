// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gdmbuilder

import (
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/optimizers"
	"github.com/nlpodyssey/spago/optimizers/adagrad"
	"github.com/nlpodyssey/spago/optimizers/adam"
	"github.com/nlpodyssey/spago/optimizers/lamb"
	"github.com/nlpodyssey/spago/optimizers/radam"
	"github.com/nlpodyssey/spago/optimizers/rmsprop"
	"github.com/nlpodyssey/spago/optimizers/sgd"
)

// NewMethod returns a new optimizers.Strategy, chosen and initialized according to
// the given config.
// It panics if the config type is unknown or unsupported.
func NewMethod[T float.DType](config optimizers.StrategyConfig) optimizers.Strategy {
	switch config := config.(type) {
	case adagrad.Config:
		return adagrad.New[T](config)
	case adam.Config:
		return adam.New[T](config)
	case radam.Config:
		return radam.New[T](config)
	case rmsprop.Config:
		return rmsprop.New[T](config)
	case lamb.Config:
		return lamb.New[T](config)
	case sgd.Config:
		return sgd.New[T](config)
	default:
		panic("gd: unknown method configuration")
	}
}
