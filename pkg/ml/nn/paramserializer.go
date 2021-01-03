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
	s.value = data.Value
	s.payload = data.Payload
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
