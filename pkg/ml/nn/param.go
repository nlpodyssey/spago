// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/utils/kvdb"
	"log"
	"sync"
)

// Param is the interface for a Model parameter.
type Param interface {
	ag.Node // it implies fn.Operand and ag.GradValue too

	// Name returns the params name (can be empty string).
	Name() string
	// SetName set the params name (can be empty string).
	SetName(name string)
	// Type returns the params type (weights, biases, undefined).
	Type() ParamsType
	// SetType set the params type (weights, biases, undefined).
	SetType(pType ParamsType)
	// SetRequiresGrad set whether the param requires gradient, or not.
	SetRequiresGrad(value bool)
	// ReplaceValue replaces the value of the parameter and clears the support structure.
	ReplaceValue(value mat.Matrix)
	// ApplyDelta updates the value of the underlying storage applying the delta.
	ApplyDelta(delta mat.Matrix)
	// Payload returns the optimizer support structure (can be nil).
	Payload() *Payload
	// SetPayload is a thread safe operation to set the given Payload on the
	// receiver Param.
	SetPayload(payload *Payload)
	// ClearPayload clears the support structure.
	ClearPayload()
}

// Params extends a slice of Param with Nodes() method.
type Params []Param

// Nodes converts the slice of Param into a slice of ag.Node.
func (ps Params) Nodes() []ag.Node {
	ns := make([]ag.Node, len(ps))
	for i, p := range ps {
		ns[i] = p
	}
	return ns
}

var _ Param = &param{}

type param struct {
	name         string
	pType        ParamsType // lazy initialization
	mu           sync.Mutex // to avoid data race
	value        mat.Matrix // store the results of a forward evaluation.
	grad         mat.Matrix // TODO: support of sparse gradients
	payload      *Payload   // additional data used for example by gradient-descend optimization methods
	hasGrad      bool
	requiresGrad bool
	storage      *kvdb.KeyValueDB // default nil
}

// ParamOption allows to configure a new Param with your specific needs.
type ParamOption func(*param)

// RequiresGrad is an option to specify whether a Param should be trained or not.
func RequiresGrad(value bool) ParamOption {
	return func(p *param) {
		p.requiresGrad = value
	}
}

// SetStorage is an option to specify a kvdb.KeyValueDB storage.
// This is useful, for example, for a memory-efficient embeddings
// Param implementation.
func SetStorage(storage *kvdb.KeyValueDB) ParamOption {
	return func(p *param) {
		p.storage = storage
	}
}

// NewParam returns a new param.
func NewParam(value mat.Matrix, opts ...ParamOption) Param {
	p := &param{
		name:         "",        // lazy initialization
		pType:        Undefined, // lazy initialization
		value:        value,
		grad:         nil, // lazy initialization
		hasGrad:      false,
		requiresGrad: true, // true by default, can be modified with the options
		payload:      nil,  // lazy initialization
		storage:      nil,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// SetName set the params name (can be empty string).
func (r *param) SetName(name string) {
	r.name = name
}

// SetType set the params type (weights, biases, undefined).
func (r *param) SetType(pType ParamsType) {
	r.pType = pType
}

// Name returns the params name (can be empty string).
func (r *param) Name() string {
	return r.name
}

// Type returns the params type (weights, biases, undefined).
func (r *param) Type() ParamsType {
	return r.pType
}

// Value returns the value of the delegate itself.
func (r *param) Value() mat.Matrix {
	return r.value
}

// ReplaceValue replaces the value of the parameter and clears the support structure.
func (r *param) ReplaceValue(value mat.Matrix) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.value = value
	r.payload = nil
	if r.storage != nil {
		r.updateStorage()
	}
}

// ScalarValue returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (r *param) ScalarValue() mat.Float {
	return r.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (r *param) Grad() mat.Matrix {
	return r.grad
}

// PropagateGrad accumulate the gradients
func (r *param) PropagateGrad(grad mat.Matrix) {
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
func (r *param) HasGrad() bool {
	return r.hasGrad
}

// RequiresGrad returns true if the param requires gradients.
func (r *param) RequiresGrad() bool {
	return r.requiresGrad
}

// SetRequiresGrad is an option to specify whether a Param should be trained or not.
func (r *param) SetRequiresGrad(value bool) {
	r.requiresGrad = value
}

// ZeroGrad clears the gradients.
func (r *param) ZeroGrad() {
	if r.grad == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	defer mat.ReleaseMatrix(r.grad) //  release memory
	r.grad = nil
	r.hasGrad = false
}

// ApplyDelta updates the value of the underlying storage applying the delta.
func (r *param) ApplyDelta(delta mat.Matrix) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Value().SubInPlace(delta)
	if r.storage != nil {
		r.updateStorage()
	}
}

// Payload returns the optimizer support structure (can be nil).
func (r *param) Payload() *Payload {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.payload
}

// SetPayload is a thread safe operation to set the given Payload on the
// receiver Param.
func (r *param) SetPayload(payload *Payload) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.payload = payload
	if r.storage != nil {
		r.updateStorage()
	}
}

// ClearPayload clears the support structure.
func (r *param) ClearPayload() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.payload = nil
	if r.storage != nil {
		r.updateStorage()
	}
}

func (r *param) updateStorage() {
	if r.storage == nil {
		return
	}

	buf := new(bytes.Buffer)
	err := MarshalBinaryParam(r, buf)
	if err != nil {
		log.Fatal(err)
	}

	err = r.storage.Put([]byte(r.name), buf.Bytes())
	if err != nil {
		log.Fatal(err)
	}
}

// Graph returns always nil since the "pure" parameter is not associated with any graph.
func (r *param) Graph() *ag.Graph {
	return nil
}

// ID returns always -1 since the "pure" parameter is not associated with any graph.
func (r *param) ID() int {
	return -1
}

// TimeStep returns always 0 since the "pure" parameter is not associated with any graph.
func (r *param) TimeStep() int {
	return 0
}

// wrappedParam returns a new wrappedParam from the param itself.
func (r *param) wrappedParam(g *ag.Graph) *wrappedParam {
	if r.requiresGrad {
		return &wrappedParam{param: r, Node: g.NewWrap(r)}
	}
	return &wrappedParam{param: r, Node: g.NewWrapNoGrad(r)}
}

var _ Param = &wrappedParam{}

// wrappedParam enriches a Param with a Node.
type wrappedParam struct {
	*param
	Node ag.Node
}

// ID dispatches the call to the Node.
func (r *wrappedParam) ID() int {
	return r.Node.ID()
}

// Graph dispatches the call to the Node.
func (r *wrappedParam) Graph() *ag.Graph {
	return r.Node.Graph()
}

// Grad dispatches the call to the Node.
func (r *wrappedParam) Grad() mat.Matrix {
	return r.Node.Grad()
}

// PropagateGrad dispatches the call to the Node.
func (r *wrappedParam) PropagateGrad(gx mat.Matrix) {
	r.Node.PropagateGrad(gx)
}

// HasGrad dispatches the call to the Node.
func (r *wrappedParam) HasGrad() bool {
	return r.Node.HasGrad()
}

// RequiresGrad dispatches the call to the Node.
func (r *wrappedParam) RequiresGrad() bool {
	return r.Node.RequiresGrad()
}

// ZeroGrad dispatches the call to the Node.
func (r *wrappedParam) ZeroGrad() {
	r.Node.ZeroGrad()
}
