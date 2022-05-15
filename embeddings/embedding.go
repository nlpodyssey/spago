// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

// Embedding is an implementation of nn.Param representing embedding values.
type Embedding[K Key] struct {
	model *Model[K]
	key   K
}

// Value satisfies the interfaces nn.Param and ag.Node.
func (e *Embedding[_]) Value() mat.Matrix {
	sd := new(storeData)
	exists, err := e.model.Store.Get(encodeKey(e.key), sd)
	if err != nil {
		panic(err)
	}
	if !exists {
		return nil
	}
	return sd.Value()
}

// ScalarValue satisfies the interfaces nn.Param and ag.Node.
func (e *Embedding[_]) ScalarValue() mat.FloatInterface {
	v := e.Value()
	if v == nil {
		panic("embeddings: cannot get scalar value from nil Matrix")
	}
	return v.Scalar()
}

// Grad satisfies the interfaces nn.Param and ag.Node.
func (e *Embedding[_]) Grad() mat.Matrix {
	grad, exists := e.model.getGrad(e.key)
	if !exists {
		return nil
	}
	return grad
}

// HasGrad satisfies the interfaces nn.Param and ag.Node.
func (e *Embedding[_]) HasGrad() bool {
	_, exists := e.model.getGrad(e.key)
	return exists
}

// RequiresGrad satisfies the interfaces nn.Param and ag.Node.
// It returns the same value of Config.Trainable of the Model tied to this
// Embedding.
func (e *Embedding[_]) RequiresGrad() bool {
	return e.model.Trainable
}

// AccGrad satisfies the interfaces nn.Param and ag.Node.
func (e *Embedding[_]) AccGrad(gx mat.Matrix) {
	e.model.accGrad(e, gx)
}

// ZeroGrad satisfies the interfaces nn.Param and ag.Node.
func (e *Embedding[_]) ZeroGrad() {
	e.model.zeroGrad(e.key)
}

// Name satisfies the interface nn.Param.
func (e *Embedding[_]) Name() string {
	switch k := any(e.key).(type) {
	case string:
		// For a string key, the param name is the string itself.
		return k
	case []byte:
		// For a []byte key, the param name is the hex representation.
		return fmt.Sprintf("%X", k)
	case int:
		// For an int key, we format the integer value.
		return fmt.Sprintf("%d", k)
	default:
		panic(fmt.Errorf("embeddings: unexpected key type %T", k))
	}
}

// Type satisfies the interface nn.Param.
// Embedding params are always considered nn.Weights.
func (e *Embedding[_]) Type() nn.ParamsType {
	return nn.Weights
}

// SetRequiresGrad satisfies the interface nn.Param.
// It always panics: it's not possible to assign a custom grad requirement
// to an Embedding parameter.
func (e *Embedding[_]) SetRequiresGrad(bool) {
	panic("embeddings: setting grad requirement on an Embedding param is not permitted")
}

// ReplaceValue satisfies the interface nn.Param.
func (e *Embedding[_]) ReplaceValue(value mat.Matrix) {
	e.model.zeroGrad(e.key)

	// Start with a new storeData, so that any
	// pre-existing payload is also cleared.
	sd := new(storeData)
	sd.SetValue(value)
	err := e.model.Store.Put(encodeKey(e.key), sd)
	if err != nil {
		panic(err)
	}
}

// ApplyDelta satisfies the interface nn.Param.
func (e *Embedding[_]) ApplyDelta(delta mat.Matrix) {
	sd := new(storeData)
	key := encodeKey(e.key)

	exists, err := e.model.Store.Get(key, sd)
	if err != nil {
		panic(err)
	}
	if !exists {
		panic(fmt.Errorf("embeddings: cannot apply delta: embedding %#v not found", e.key))
	}
	if sd.Value() == nil {
		panic(fmt.Errorf("embeddings: cannot apply delta: value of embedding %#v is nil", e.key))
	}

	sd.Value().SubInPlace(delta)
	err = e.model.Store.Put(key, sd)
	if err != nil {
		panic(err)
	}
}

// Payload satisfies the interface nn.Param.
func (e *Embedding[_]) Payload() *nn.Payload {
	sd := new(storeData)
	exists, err := e.model.Store.Get(encodeKey(e.key), sd)
	if err != nil {
		panic(err)
	}
	if !exists {
		return nil
	}
	return sd.Payload()
}

// SetPayload satisfies the interface nn.Param.
func (e *Embedding[_]) SetPayload(payload *nn.Payload) {
	sd := new(storeData)
	key := encodeKey(e.key)

	// Ignore whether a key/value already exists: if not, we simply start
	// with a valid storeData zero-value.
	_, err := e.model.Store.Get(key, sd)
	if err != nil {
		panic(err)
	}
	sd.SetPayload(payload)
	err = e.model.Store.Put(key, sd)
	if err != nil {
		panic(err)
	}
}

// ClearPayload satisfies the interface nn.Param.
func (e *Embedding[_]) ClearPayload() {
	sd := new(storeData)
	key := encodeKey(e.key)

	// Ignore whether a key/value already exists: if not, we simply start
	// with a valid storeData zero-value.
	_, err := e.model.Store.Get(key, sd)
	if err != nil {
		panic(err)
	}
	sd.SetPayload(nil)
	err = e.model.Store.Put(key, sd)
	if err != nil {
		panic(err)
	}
}
