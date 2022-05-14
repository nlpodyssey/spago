// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"encoding"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	_ encoding.BinaryMarshaler   = &storeData[float32]{}
	_ encoding.BinaryUnmarshaler = &storeData[float32]{}
)

func TestStoreData(t *testing.T) {
	sd := &storeData[float32]{}

	assert.Nil(t, sd.Value())
	assert.Nil(t, sd.Payload())
	assertMarshalUnmarshalProducesSameValue(t, sd)

	// Set Value

	value := mat.NewScalar[float32](42)
	sd.SetValue(value)

	assert.Same(t, value, sd.Value())
	assert.Nil(t, sd.Payload())
	assertMarshalUnmarshalProducesSameValue(t, sd)

	// Set Payload

	payload := &nn.Payload[float32]{
		Label: 123,
		Data: []mat.Matrix{
			mat.NewScalar[float32](11),
			mat.NewScalar[float32](22),
		},
	}
	sd.SetPayload(payload)

	assert.Same(t, value, sd.Value())
	assert.Same(t, payload, sd.Payload())
	assertMarshalUnmarshalProducesSameValue(t, sd)

	// Set Value to nil

	sd.SetValue(nil)

	assert.Nil(t, sd.Value())
	assert.Same(t, payload, sd.Payload())
	assertMarshalUnmarshalProducesSameValue(t, sd)

	// Set Payload to nil

	sd.SetPayload(nil)

	assert.Nil(t, sd.Value())
	assert.Nil(t, sd.Payload())
	assertMarshalUnmarshalProducesSameValue(t, sd)

}

func TestStoreData_MarshalBinary(t *testing.T) {
	value := mat.NewScalar[float32](42)
	payload := &nn.Payload[float32]{
		Label: 123,
		Data: []mat.Matrix{
			mat.NewScalar[float32](11),
			mat.NewScalar[float32](22),
		},
	}

	sd := marshalUnmarshal(t, &storeData[float32]{
		value:   value,
		payload: payload,
	})

	// After unmarshaling, value and payload are nil, while their
	// marshaled data is present
	assert.Nil(t, sd.value)
	assert.Nil(t, sd.payload)
	assert.NotNil(t, sd.marshaledValue)
	assert.NotNil(t, sd.marshaledPayload)

	{ // Marshaling doesn't change the current field values
		other := marshalUnmarshal(t, sd)
		assertMatrixEqual(t, value, other.Value())
		assertPayloadEqual(t, payload, other.Payload())

		assert.Nil(t, sd.value)
		assert.Nil(t, sd.payload)
		assert.NotNil(t, sd.marshaledValue)
		assert.NotNil(t, sd.marshaledPayload)
	}

	// Value() unmarshals the value and clears the value-bytes
	v := sd.Value()
	assertMatrixEqual(t, value, v)

	assert.Same(t, v, sd.value)
	assert.Nil(t, sd.payload)
	assert.Nil(t, sd.marshaledValue)
	assert.NotNil(t, sd.marshaledPayload)

	{ // Marshaling doesn't change the current field values
		other := marshalUnmarshal(t, sd)
		assertMatrixEqual(t, value, other.Value())
		assertPayloadEqual(t, payload, other.Payload())

		assert.Same(t, v, sd.value)
		assert.Nil(t, sd.payload)
		assert.Nil(t, sd.marshaledValue)
		assert.NotNil(t, sd.marshaledPayload)
	}

	// Payload() unmarshals the payload and clears the payload-bytes
	p := sd.Payload()
	assertPayloadEqual(t, payload, p)

	assert.Same(t, v, sd.value)
	assert.Same(t, p, sd.payload)
	assert.Nil(t, sd.marshaledValue)
	assert.Nil(t, sd.marshaledPayload)

	{ // Marshaling doesn't change the current field values
		other := marshalUnmarshal(t, sd)
		assertMatrixEqual(t, value, other.Value())
		assertPayloadEqual(t, payload, other.Payload())

		assert.Same(t, v, sd.value)
		assert.Same(t, p, sd.payload)
		assert.Nil(t, sd.marshaledValue)
		assert.Nil(t, sd.marshaledPayload)
	}
}

func assertMarshalUnmarshalProducesSameValue(t *testing.T, sd *storeData[float32]) {
	t.Helper()
	other := marshalUnmarshal(t, sd)
	assertStoreDataEqual(t, sd, other)
}

func marshalUnmarshal(t *testing.T, sd *storeData[float32]) *storeData[float32] {
	t.Helper()

	data, err := sd.MarshalBinary()
	require.NoError(t, err)

	other := new(storeData[float32])
	err = other.UnmarshalBinary(data)
	require.NoError(t, err)
	return other
}

func assertStoreDataEqual(t *testing.T, expected, actual *storeData[float32]) {
	t.Helper()
	assertMatrixEqual(t, expected.Value(), actual.Value())
	assertPayloadEqual(t, expected.Payload(), actual.Payload())
}

func assertMatrixEqual(t *testing.T, expected, actual mat.Matrix) {
	t.Helper()

	if expected == nil {
		assert.Nil(t, actual)
		return
	}

	assert.NotNil(t, actual)
	if actual == nil {
		return
	}
	assert.Equal(t, expected.Rows(), actual.Rows())
	assert.Equal(t, expected.Columns(), actual.Columns())
	assert.Equal(t, expected.Data(), actual.Data())
}

func assertPayloadEqual(t *testing.T, expected, actual *nn.Payload[float32]) {
	t.Helper()

	if expected == nil {
		assert.Nil(t, actual)
		return
	}

	assert.NotNil(t, actual)
	if actual == nil {
		return
	}

	assert.Equal(t, expected.Label, actual.Label)
	assert.Len(t, actual.Data, len(expected.Data))
	if len(actual.Data) != len(expected.Data) {
		return
	}
	for i := range expected.Data {
		assertMatrixEqual(t, expected.Data[i], actual.Data[i])
	}
}
