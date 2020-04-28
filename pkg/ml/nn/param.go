// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"strings"
	"sync"
)

type ParamsType int

const (
	Weights ParamsType = iota
	Biases
	Undefined
)

var pts = []ParamsType{Weights, Biases, Undefined}

func (t ParamsType) String() string {
	return [...]string{"weights", "biases", "undefined"}[t] // important lower case
}

// ToType convert a string to a ParamsType. It returns Undefined if the string doesn't match any ParamsType.
func ToType(s string) ParamsType {
	for _, item := range pts {
		if item.String() == strings.ToLower(s) {
			return item
		}
	}
	return Undefined
}

var (
	_ fn.Operand   = &Param{}
	_ ag.GradValue = &Param{}
)

type Param struct {
	name         string
	pType        ParamsType  // lazy initialization
	mu           sync.Mutex  // to avoid data race
	value        mat.Matrix  // store the results of a forward evaluation.
	grad         mat.Matrix  // TODO: support of sparse gradients
	support      *gd.Support // additional data used by the gradient-descend optimization methods
	hasGrad      bool
	requiresGrad bool
}

// NewParam returns a new param.
func NewParam(value mat.Matrix) *Param {
	return &Param{
		name:         "",        // lazy initialization
		pType:        Undefined, // lazy initialization
		value:        value,
		grad:         nil, // lazy initialization
		hasGrad:      false,
		requiresGrad: true, // TODO: might not always have to be true?
		support:      nil,  // lazy initialization
	}
}

// SetName set the params name (can be empty string).
func (r *Param) SetName(name string) {
	r.name = name
}

// SetType set the params type (weights, biases, undefined).
func (r *Param) SetType(name string) {
	r.name = name
}

// Name returns the params name (can be empty string).
func (r *Param) Name() string {
	return r.name
}

// Type returns the params type (weights, biases, undefined).
func (r *Param) Type() ParamsType {
	return r.pType
}

// Value returns the value of the delegate itself.
func (r *Param) Value() mat.Matrix {
	return r.value
}

// ReplaceValue replaces the value of the parameter and clears the support structure.
func (r *Param) ReplaceValue(value mat.Matrix) {
	r.value = value
	r.ClearSupport()
}

// ScalarValue() returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (r *Param) ScalarValue() float64 {
	return r.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Param) Grad() mat.Matrix {
	return r.grad
}

// PropagateGrad accumulate the gradients
func (r *Param) PropagateGrad(grad mat.Matrix) {
	if !r.requiresGrad {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.grad == nil {
		r.grad = mat.GetEmptyDenseWorkspace(r.value.Dims()) // this could reduce the number of allocations
	}
	r.grad.AddInPlace(grad)
	r.hasGrad = true
}

// HasGrad returns true if there are accumulated gradients.
func (r *Param) HasGrad() bool {
	return r.hasGrad
}

// RequiresGrad returns true if the param requires gradients.
func (r *Param) RequiresGrad() bool {
	return r.requiresGrad
}

func (r *Param) SetRequiresGrad(requiresGrad bool) {
	r.requiresGrad = requiresGrad
	if !r.requiresGrad && r.hasGrad {
		r.ZeroGrad()
	}
}

// ZeroGrad clears the gradients.
func (r *Param) ZeroGrad() {
	if r.grad == nil {
		return
	}
	defer mat.ReleaseDense(r.grad.(*mat.Dense)) // release memory
	r.grad = nil
	r.hasGrad = false
}

// ApplyDelta updates the value of the underlying storage applying the delta.
func (r *Param) ApplyDelta(delta mat.Matrix) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Value().SubInPlace(delta)
}

// Support returns the optimizer support structure (can be nil).
func (r *Param) Support() *gd.Support {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.support
}

// SetSupport sets the optimizer support structure.
// Use ClearSupport() to set a nil support.
func (r *Param) SetSupport(supp *gd.Support) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.support = supp
}

func (r *Param) GetOrSetSupport(m gd.OptimizationMethod) *gd.Support {
	if r.Support() == nil || r.Support().Name == gd.None {
		r.SetSupport(m.NewSupport(r.Value().Dims()))
	} else if r.Support().Name != m.Name() {
		panic("gd: support structure non compatible with the optimization method")
	}
	return r.Support()
}

// ClearSupport clears the support structure.
func (r *Param) ClearSupport() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.support = nil
}
