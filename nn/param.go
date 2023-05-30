// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import "github.com/nlpodyssey/spago/mat"

type Param struct {
	mat.Matrix
	State interface{} // support structure for the optimization algorithm
}

// NewParam returns a new param.
func NewParam(value mat.Matrix) *Param {
	if value == nil {
		panic("nn: cannot create a new param with a nil value")
	}
	p := &Param{
		Matrix: value.Clone(),
		State:  nil,
	}
	p.SetRequiresGrad(true)
	return p
}

// WithGrad sets whether the param requires gradients (default true)
func (p *Param) WithGrad(value bool) *Param {
	p.SetRequiresGrad(value)
	return p
}

func (p *Param) ReplaceValue(value mat.Matrix) {
	p.Matrix = value
	p.State = nil
}
