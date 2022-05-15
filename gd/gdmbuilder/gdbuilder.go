// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gdmbuilder

import (
	"github.com/nlpodyssey/spago/gd"
	"github.com/nlpodyssey/spago/gd/adagrad"
	"github.com/nlpodyssey/spago/gd/adam"
	"github.com/nlpodyssey/spago/gd/lamb"
	"github.com/nlpodyssey/spago/gd/radam"
	"github.com/nlpodyssey/spago/gd/rmsprop"
	"github.com/nlpodyssey/spago/gd/sgd"
	"github.com/nlpodyssey/spago/mat"
)

// NewMethod returns a new gd.Method, chosen and initialized according to
// the given config.
// It panics if the config type is unknown or unsupported.
func NewMethod[T mat.DType](config gd.MethodConfig) gd.Method {
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
