// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/binary"
	"errors"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/utils"
	"io"
)

// ParamSerializer allows serialization and deserialization of a single Param.
type ParamSerializer struct {
	*param
}

// NewParamSerializer returns a new ParamSerializer.
// It returns an error if the Param doesn't support serialization.
func NewParamSerializer(p Param) (*ParamSerializer, error) {
	switch p := p.(type) {
	case *param:
		return &ParamSerializer{param: p}, nil
	default:
		return nil, errors.New("nn: param type not supported for serialization")
	}
}

// Serialize dumps the Param to the writer.
func (s *ParamSerializer) Serialize(w io.Writer) error {
	return paramDataMarshalBinaryTo(&paramData{
		Value:   s.value.(*mat.Dense),
		Payload: s.payload,
	}, w)
}

// Deserialize assigns reads a Param the reader.
func (s *ParamSerializer) Deserialize(r io.Reader) error {
	data, err := paramDataUnmarshalBinaryFrom(r)
	if err != nil {
		return err
	}
	s.value = data.Value
	s.payload = data.Payload
	return err
}

type paramData struct {
	Value   *mat.Dense
	Payload *Payload
}

func paramDataMarshalBinaryTo(data *paramData, w io.Writer) error {
	err := mat.MarshalBinaryTo(data.Value, w)
	if err != nil {
		return err
	}
	err = PayloadMarshalBinaryTo(data.Payload, w)
	if err != nil {
		return err
	}
	return err
}

func paramDataUnmarshalBinaryFrom(r io.Reader) (*paramData, error) {
	value, err := mat.NewUnmarshalBinaryFrom(r)
	if err != nil {
		return nil, err
	}
	supp, err := NewPayloadUnmarshalBinaryFrom(r)
	if err != nil {
		return nil, err
	}
	return &paramData{Value: value, Payload: supp}, err
}

// PayloadMarshalBinaryTo returns the number of bytes written into w and an error, if any.
func PayloadMarshalBinaryTo(supp *Payload, w io.Writer) error {
	h := header{Label: int64(supp.Label), Size: int64(len(supp.Data))}
	err := h.marshalBinaryTo(w)
	if err != nil {
		return err
	}
	err = mat.MarshalBinarySlice(supp.Data, w)
	return err
}

// NewPayloadUnmarshalBinaryFrom reads a Payload from the given reader.
func NewPayloadUnmarshalBinaryFrom(r io.Reader) (*Payload, error) {
	var h header
	err := h.unmarshalBinaryFrom(r)
	if err != nil {
		return nil, err
	}
	data := make([]mat.Matrix, h.Size)
	err = mat.NewUnmarshalBinarySlice(data, r)
	if err != nil {
		return nil, err
	}
	supp := &Payload{
		Label: int(h.Label),
		Data:  data,
	}
	return supp, err
}

type header struct {
	Label int64
	Size  int64
}

var headerSize = binary.Size(header{})

func (s header) marshalBinaryTo(w io.Writer) error {
	buf := bytes.NewBuffer(make([]byte, 0, headerSize))
	err := binary.Write(buf, binary.LittleEndian, s)
	if err != nil {
		return err
	}
	_, err = w.Write(buf.Bytes())
	return err
}

func (s *header) unmarshalBinary(buf []byte) error {
	err := binary.Read(bytes.NewReader(buf), binary.LittleEndian, s)
	if err != nil {
		return err
	}
	return nil
}

func (s *header) unmarshalBinaryFrom(r io.Reader) error {
	buf := make([]byte, headerSize)
	n, err := utils.ReadFull(r, buf)
	if err != nil {
		return err
	}
	return s.unmarshalBinary(buf[:n])
}
