// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"bytes"
	"encoding/binary"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

// storeData allows efficient handling of an embedding's core data, that ease
// a pair of value and payload, in respect to their insertion and reading
// to and from a store.Store.
//
// Some store implementations don't require any sort of data serialization,
// for example the built-in memstore.Store. In this case, the object is
// directly inserted in the store and there's no need for special handling: the
// functions to get and set value and payload operate directly on the
// corresponding fields.
//
// Other store implementation might instead require data to be serialized
// (marshaled), for example the built-in diskstore.Store. The Embedding
// type allows to read and write the value and the payload only individually.
// Each time one of these element is modified, both of them must be saved
// (and serialized) again, and each time one of them is read, the data of both
// elements is read (and should be deserialized) from the store.
// In order to keep these reading and writing operations efficient, this struct
// provides a lazy unmarshaling mechanism. The unmarshaling function simply
// stores the raw bytes of value and payload into marshaledValue and
// marshaledPayload. The actual decoding of each of these elements happens only
// when the effective data value is needed. When the same object must be
// re-encoded again, the raw bytes of the element that was not requested or
// modified are already there.
//
// For example, if a storeData object is read from the store, the
// unmarshaling only splits and stores the raw-byte data values. As soon as
// Value() is requested, it is decoded once; subsequent calls to Value() will
// always return the same deserialized value. The value can also be modified
// with SetValue(). When the storeData object must be saved again into the
// store, the value is encoded, while the payload (that never accessed nor
// modified) is still in its original raw-bite form, which is used as-is as
// part of the final encoded data. In this way, there has been no need to
// decode nor re-encode the payload.
type storeData[T mat.DType] struct {
	value            mat.Matrix[T]
	payload          *nn.Payload[T]
	marshaledValue   []byte
	marshaledPayload []byte
}

// Value returns the mat.Matrix value, which is lazily decoded once if necessary.
func (sd *storeData[T]) Value() mat.Matrix[T] {
	if sd.marshaledValue != nil {
		m, err := mat.UnmarshalBinaryMatrix[T](bytes.NewReader(sd.marshaledValue))
		if err != nil {
			panic(err)
		}
		sd.marshaledValue = nil
		sd.value = m
	}
	return sd.value
}

// Payload returns the nn.Payload value, which is lazily decoded once if necessary.
func (sd *storeData[T]) Payload() *nn.Payload[T] {
	if sd.marshaledPayload != nil {
		if len(sd.marshaledPayload) > 0 {
			sd.payload = new(nn.Payload[T])
			if err := sd.payload.UnmarshalBinary(sd.marshaledPayload); err != nil {
				panic(err)
			}
		}
		sd.marshaledPayload = nil
	}
	return sd.payload
}

// SetValue sets the mat.Matrix value. If a previously unmarshaled value's raw
// data is present, it is invalidated (removed).
func (sd *storeData[T]) SetValue(v mat.Matrix[T]) {
	sd.marshaledValue = nil
	sd.value = v
}

// SetPayload sets the nn.Payload value. If a previously unmarshaled payload's
// raw data is present, it is invalidated (removed).
func (sd *storeData[T]) SetPayload(v *nn.Payload[T]) {
	sd.marshaledPayload = nil
	sd.payload = v
}

// MarshalBinary satisfies encoding.BinaryMarshaler interface.
func (sd *storeData[T]) MarshalBinary() ([]byte, error) {
	valueBytes, err := sd.marshalValue()
	if err != nil {
		return nil, err
	}

	payloadBytes, err := sd.marshalPayload()
	if err != nil {
		return nil, err
	}

	data := make([]byte, 8+len(valueBytes)+len(payloadBytes))

	// Layout:
	//   - 8 bytes: payload offset (uint64 LE)
	//   - len(valueBytes) bytes: marshaled value
	//   - len(payloadBytes) bytes: marshaled payload
	payloadOffset := 8 + len(valueBytes)
	binary.LittleEndian.PutUint64(data[:8], uint64(payloadOffset))
	copy(data[8:payloadOffset], valueBytes)
	copy(data[payloadOffset:], payloadBytes)

	return data, nil
}

// UnmarshalBinary satisfies encoding.BinaryUnmarshaler interface.
func (sd *storeData[T]) UnmarshalBinary(data []byte) (err error) {
	payloadOffset := int(binary.LittleEndian.Uint64(data[:8]))
	sd.marshaledValue = data[8:payloadOffset]
	sd.marshaledPayload = data[payloadOffset:]
	sd.value = nil
	sd.payload = nil
	return nil
}

func (sd *storeData[T]) marshalValue() ([]byte, error) {
	if sd.marshaledValue != nil {
		return sd.marshaledValue, nil
	}

	var buf bytes.Buffer
	if err := mat.MarshalBinaryMatrix(sd.value, &buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (sd *storeData[T]) marshalPayload() ([]byte, error) {
	if sd.marshaledPayload != nil {
		return sd.marshaledPayload, nil
	}

	if sd.payload == nil {
		return make([]byte, 0), nil
	}
	return sd.payload.MarshalBinary()
}
