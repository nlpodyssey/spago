// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gdmbuilder

import (
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/optimizer"
	"github.com/nlpodyssey/spago/optimizer/adagrad"
	"github.com/nlpodyssey/spago/optimizer/adam"
	"github.com/nlpodyssey/spago/optimizer/lamb"
	"github.com/nlpodyssey/spago/optimizer/radam"
	"github.com/nlpodyssey/spago/optimizer/rmsprop"
	"github.com/nlpodyssey/spago/optimizer/sgd"
)

// NewMethod returns a new optimizer.Strategy, chosen and initialized according to
// the given config.
// It panics if the config type is unknown or unsupported.
func NewMethod[T float.DType](config optimizer.StrategyConfig) optimizer.Strategy {
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
