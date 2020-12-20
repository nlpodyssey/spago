// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/binary"
	"io"
	"log"
	"strings"
	"sync"

	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/pkg/utils/kvdb"
)

// ParamsType is the enumeration-like type used for the set of parameter
// (Param) types of a neural network Model.
type ParamsType int

const (
	// Weights identifies a Param containing weights.
	Weights ParamsType = iota
	// Biases identifies a Param containing biases.
	Biases
	// Undefined identifies a generic Param, which cannot be described
	// with other ParamsType values.
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

// Payload contains the support data used for example by the optimization methods
type Payload struct {
	Label int
	Data  []mat.Matrix
}

// NewEmptySupport returns an empty support structure, not connected to any optimization method.
func NewEmptySupport() *Payload {
	return &Payload{
		Label: 0, // important set the label to zero
		Data:  make([]mat.Matrix, 0),
	}
}

// Param is a parameter of a Model.
type Param struct {
	name         string
	pType        ParamsType // lazy initialization
	mu           sync.Mutex // to avoid data race
	value        mat.Matrix // store the results of a forward evaluation.
	grad         mat.Matrix // TODO: support of sparse gradients
	payload      *Payload   // additional data used for example by gradient-descend optimization methods
	hasGrad      bool
	requiresGrad bool
	storage      kvdb.KeyValueDB // default nil
}

// ParamOption allows to configure a new Param with your specific needs.
type ParamOption func(*Param)

// RequiresGrad is an option to specify whether a Param should be trained or not.
func RequiresGrad(value bool) ParamOption {
	return func(p *Param) {
		p.requiresGrad = value
	}
}

// SetStorage is an option to specify a kvdb.KeyValueDB storage.
// This is useful, for example, for a memory-efficient embeddings
// Param implementation.
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
		payload:      nil,  // lazy initialization
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
	r.payload = nil
	if r.storage != nil {
		r.updateStorage()
	}
}

// ScalarValue returns the the scalar value of the node.
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

// Payload returns the optimizer support structure (can be nil).
func (r *Param) Payload() *Payload {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.payload
}

// SetPayload is a thread safe operation to set the given Payload on the
// receiver Param.
func (r *Param) SetPayload(payload *Payload) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.payload = payload
	if r.storage != nil {
		r.updateStorage()
	}
}

// ClearPayload clears the support structure.
func (r *Param) ClearPayload() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.payload = nil
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

// MarshalBinary satisfies package pkg/encoding/gob custom marshaling interface
func (r *Param) MarshalBinary() ([]byte, error) {
	var b bytes.Buffer
	_, err := mat.MarshalBinaryTo(r.value, &b)
	if err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// UnmarshalBinary satisfies pkg/encoding/gob custom marshaling interface
func (r *Param) UnmarshalBinary(data []byte) error {
	b := bytes.NewBuffer(data)
	value, _, err := mat.NewUnmarshalBinaryFrom(b)
	r.value = value
	return err
}

// ParamSerializer allows serialization and deserialization of a single Param.
type ParamSerializer struct {
	*Param
}

// Serialize dumps the Param to the writer.
func (s *ParamSerializer) Serialize(w io.Writer) (int, error) {
	return paramDataMarshalBinaryTo(&paramData{
		Value:   s.value.(*mat.Dense),
		Payload: s.payload,
	}, w)
}

// Deserialize assigns reads a Param the reader.
func (s *ParamSerializer) Deserialize(r io.Reader) (n int, err error) {
	var data *paramData
	data, n, err = paramDataUnmarshalBinaryFrom(r)
	if err != nil {
		return
	}
	s.Param.value = data.Value
	s.Param.payload = data.Payload
	return
}

type paramData struct {
	Value   *mat.Dense
	Payload *Payload
}

func paramDataMarshalBinaryTo(data *paramData, w io.Writer) (int, error) {
	n, err := mat.MarshalBinaryTo(data.Value, w)
	if err != nil {
		return n, err
	}
	n2, err := PayloadMarshalBinaryTo(data.Payload, w)
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
	supp, n2, err := NewPayloadUnmarshalBinaryFrom(r)
	n += n2
	if err != nil {
		return nil, n, err
	}
	return &paramData{Value: value, Payload: supp}, n, err
}

// PayloadMarshalBinaryTo returns the number of bytes written into w and an error, if any.
func PayloadMarshalBinaryTo(supp *Payload, w io.Writer) (int, error) {
	h := header{Label: int64(supp.Label), Size: int64(len(supp.Data))}
	n, err := h.marshalBinaryTo(w)
	if err != nil {
		return n, err
	}
	nn, err := mat.MarshalBinarySlice(supp.Data, w)
	n += nn
	return n, err
}

// NewPayloadUnmarshalBinaryFrom reads a Payload from the given reader.
func NewPayloadUnmarshalBinaryFrom(r io.Reader) (*Payload, int, error) {
	var h header
	n, err := h.unmarshalBinaryFrom(r)
	if err != nil {
		return nil, n, err
	}
	data := make([]mat.Matrix, h.Size)
	nn, err := mat.NewUnmarshalBinarySlice(data, r)
	n = +nn
	if err != nil {
		return nil, n, err
	}
	supp := &Payload{
		Label: int(h.Label),
		Data:  data,
	}
	return supp, n, err
}

type header struct {
	Label int64
	Size  int64
}

var headerSize = binary.Size(header{})

func (s header) marshalBinaryTo(w io.Writer) (int, error) {
	buf := bytes.NewBuffer(make([]byte, 0, headerSize))
	err := binary.Write(buf, binary.LittleEndian, s)
	if err != nil {
		return 0, err
	}
	return w.Write(buf.Bytes())
}

func (s *header) unmarshalBinary(buf []byte) error {
	err := binary.Read(bytes.NewReader(buf), binary.LittleEndian, s)
	if err != nil {
		return err
	}
	return nil
}

func (s *header) unmarshalBinaryFrom(r io.Reader) (int, error) {
	buf := make([]byte, headerSize)
	n, err := utils.ReadFull(r, buf)
	if err != nil {
		return n, err
	}
	return n, s.unmarshalBinary(buf[:n])
}
