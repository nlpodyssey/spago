// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/utils/kvdb"
	"io"
	"log"
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
	_ fn.Operand     = &Param{}
	_ ag.GradValue   = &Param{}
	_ gd.Optimizable = &Param{}
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
	storage      kvdb.KeyValueDB // default nil
}

type ParamOption func(*Param)

func RequiresGrad(value bool) ParamOption {
	return func(p *Param) {
		p.requiresGrad = value
	}
}

func SetStorage(storage kvdb.KeyValueDB) ParamOption {
	return func(p *Param) {
		p.storage = storage
	}
}

// NewParam returns a new param.
func NewParam(value mat.Matrix, opts ...ParamOption) *Param {
	p := &Param{
		name:         "",        // lazy initialization
		pType:        Undefined, // lazy initialization
		value:        value,
		grad:         nil, // lazy initialization
		hasGrad:      false,
		requiresGrad: true, // true by default, can be modified with the options
		support:      nil,  // lazy initialization
		storage:      nil,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
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
	r.mu.Lock()
	defer r.mu.Unlock()
	r.value = value
	r.support = nil
	if r.storage != nil {
		r.updateStorage()
	}
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

// ZeroGrad clears the gradients.
func (r *Param) ZeroGrad() {
	if r.grad == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	defer mat.ReleaseDense(r.grad.(*mat.Dense)) //  release memory
	r.grad = nil
	r.hasGrad = false
}

// ApplyDelta updates the value of the underlying storage applying the delta.
func (r *Param) ApplyDelta(delta mat.Matrix) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Value().SubInPlace(delta)
	if r.storage != nil {
		r.updateStorage()
	}
}

// Support returns the optimizer support structure (can be nil).
func (r *Param) Support() *gd.Support {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.support
}

func (r *Param) SetSupport(supp *gd.Support) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.support = supp
	if r.storage != nil {
		r.updateStorage()
	}
}

func (r *Param) GetOrSetSupport(m gd.Method) *gd.Support {
	r.mu.Lock()
	defer r.mu.Unlock()
	switch {
	case r.support == nil:
		r.support = m.NewSupport(r.Value().Dims())
		r.updateStorage()
		return r.support
	case r.support.Name == gd.None:
		r.support = m.NewSupport(r.Value().Dims())
		r.updateStorage()
		return r.support
	case r.support.Name == m.Name():
		return r.support
	default:
		panic("gd: support structure non compatible with the optimization method")
	}
}

// ClearSupport clears the support structure.
func (r *Param) ClearSupport() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.support = nil
	if r.storage != nil {
		r.updateStorage()
	}
}

func (r *Param) updateStorage() {
	if r.storage == nil {
		return
	}
	var buf bytes.Buffer
	if _, err := (&ParamSerializer{Param: r}).Serialize(&buf); err != nil {
		log.Fatal(err)
	}
	if err := r.storage.Put([]byte(r.name), buf.Bytes()); err != nil {
		log.Fatal(err)
	}
}

type ParamSerializer struct {
	*Param
}

func (s *ParamSerializer) Serialize(w io.Writer) (int, error) {
	return paramDataMarshalBinaryTo(&paramData{
		Value:   s.value.(*mat.Dense),
		Support: s.support,
	}, w)
}

func (s *ParamSerializer) Deserialize(r io.Reader) (n int, err error) {
	var data *paramData
	data, n, err = paramDataUnmarshalBinaryFrom(r)
	if err != nil {
		return
	}
	s.Param.value = data.Value
	s.Param.support = data.Support
	return
}

type paramData struct {
	Value   *mat.Dense
	Support *gd.Support
}

func paramDataMarshalBinaryTo(data *paramData, w io.Writer) (int, error) {
	n, err := mat.MarshalBinaryTo(data.Value, w)
	if err != nil {
		return n, err
	}
	n2, err := gd.MarshalBinaryTo(data.Support, w)
	n += n2
	if err != nil {
		return n, err
	}
	return n, err
}

func paramDataUnmarshalBinaryFrom(r io.Reader) (*paramData, int, error) {
	value, n, err := mat.NewUnmarshalBinaryFrom(r)
	if err != nil {
		return nil, n, err
	}
	supp, n2, err := gd.NewUnmarshalBinaryFrom(r)
	n += n2
	if err != nil {
		return nil, n, err
	}
	return &paramData{Value: value, Support: supp}, n, err
}
