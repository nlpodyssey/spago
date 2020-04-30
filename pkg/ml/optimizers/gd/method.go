// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

type MethodName int

const (
	None MethodName = iota
	SGD
	AdaGrad
	Adam
	RMSProp
)

// Empty interface implemented by the configuration structures of AdaGrad, Adam, RMSProp and SGD.
type MethodConfig interface{}

// Optimization Method
type Method interface {
	Name() MethodName
	// Delta returns the difference between the current params and where the method wants it to be.
	Delta(param Optimizable) mat.Matrix
	// NewSupport returns a new support structure with the given dimensions
	NewSupport(r, c int) *Support
}
