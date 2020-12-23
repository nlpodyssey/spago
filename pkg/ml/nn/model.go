// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/pkg/errors"
	"io"
	"reflect"
)

// Model contains the serializable parameters.
type Model interface {
	// NewProc returns a new processor to execute the forward step.
	NewProc(ctx Context) Processor
}

func Contextualize(m Model, g *ag.Graph) Model {
	orig := reflect.ValueOf(m)
	if orig.Kind() == reflect.Ptr {
		orig = orig.Elem()
	}
	torig := reflect.TypeOf(m).Elem()
	dest := reflect.New(torig)

	initContextualizedModel(g, dest.Elem(), orig, torig)

	return dest.Interface().(Model)
}

func initContextualizedModel(g *ag.Graph, dest, orig reflect.Value, torig reflect.Type) {
	n := orig.NumField()
	for i := 0; i < n; i++ {
		forig, fdest := orig.Field(i), dest.Field(i)
		tag := torig.Field(i).Tag
		if tag.Get("type") == "processor" {
			continue // skip any initialization
		}
		switch item := forig.Interface().(type) {
		case Param:
			if p, ok := item.(*param); ok {
				fdest.Set(reflect.ValueOf(p.wrappedParam(g)))
			} else {
				panic(errors.New("nn: unexpected implementation of `Param` interface."))
			}
		case []Param:
			dItem := make([]Param, len(item))
			for j := 0; j < len(item); j++ {
				if p, ok := item[j].(*param); ok {
					dItem[j] = p.wrappedParam(g)
				} else {
					panic(errors.New("nn: unexpected implementation of `Param` interface."))
				}
			}
			fdest.Set(reflect.ValueOf(dItem))
		default:
			// TODO: handle slices, maps and structure of params
			fdest.Set(forig)
		}
	}
}

// ForEachParam iterate all the parameters of a model also exploring the sub-parameters recursively.
// TODO: don't loop the field every time, use a lazy initialized "params list" instead (?)
func ForEachParam(m Model, callback func(param Param)) {
	newParamsTraversal(callback, true).walk(m)
}

// ForEachParamStrict iterate all the parameters of a model without exploring the sub-models.
func ForEachParamStrict(m Model, callback func(param Param)) {
	newParamsTraversal(callback, false).walk(m)
}

// ParamsIterator is implemented by any value that has the ParamsList method,
// which should return the list of parameters of one or more models.
type ParamsIterator interface {
	ParamsList() []Param
}

var _ ParamsIterator = &DefaultParamsIterator{}

// DefaultParamsIterator is spaGO default implementation of a ParamsIterator.
type DefaultParamsIterator struct {
	models []Model
}

// NewDefaultParamsIterator returns a new DefaultParamsIterator.
func NewDefaultParamsIterator(models ...Model) *DefaultParamsIterator {
	return &DefaultParamsIterator{models: models}
}

// ParamsList returns a slice with all Param elements from all models held by
// the DefaultParamsIterator.
func (i *DefaultParamsIterator) ParamsList() []Param {
	params := make([]Param, 0)
	for _, model := range i.models {
		ForEachParam(model, func(param Param) {
			params = append(params, param)
		})
	}
	return params
}

// ZeroGrad set the gradients of all model's parameters (including sub-params) to zeros.
// TODO: use ParamsIterator?
func ZeroGrad(m Model) {
	ForEachParam(m, func(param Param) {
		param.ZeroGrad()
	})
}

// ClearSupport clears the support structure of all model's parameters (including sub-params).
// TODO: use ParamsIterator?
func ClearSupport(m Model) {
	ForEachParam(m, func(param Param) {
		param.ClearPayload()
	})
}

// DumpParamsVector dumps all params of a Model into a single Dense vector.
// TODO: use ParamsIterator?
func DumpParamsVector(model Model) *mat.Dense {
	data := make([]float64, 0)
	ForEachParam(model, func(param Param) {
		data = append(data, param.Value().Data()...)
	})
	return mat.NewVecDense(data)
}

// LoadParamsVector sets all params of a Model from a previously dumped Dense vector.
// TODO: use ParamsIterator?
func LoadParamsVector(model Model, vector *mat.Dense) {
	data := vector.Data()
	offset := 0
	ForEachParam(model, func(param Param) {
		size := param.Value().Size()
		param.Value().SetData(data[offset : offset+size])
		offset += size
	})
}

// MakeNewModels return n new models.
// The callback is delegated to return a new model for each i-item.
func MakeNewModels(n int, callback func(i int) Model) []Model {
	lst := make([]Model, n)
	for i := 0; i < n; i++ {
		lst[i] = callback(i)
	}
	return lst
}

var _ utils.Serializer = &ParamsSerializer{}
var _ utils.Deserializer = &ParamsSerializer{}

// ParamsSerializer allows serialization and deserialization of all
// parameters of a given Model.
type ParamsSerializer struct {
	Model
}

// NewParamsSerializer returns a new ParamsSerializer.
func NewParamsSerializer(m Model) *ParamsSerializer {
	return &ParamsSerializer{Model: m}
}

// Serialize dumps the params values to the writer.
// TODO: use ParamsIterator?
func (m *ParamsSerializer) Serialize(w io.Writer) (n int, err error) {
	ForEachParam(m, func(param Param) {
		cnt, err2 := mat.MarshalBinaryTo(param.Value(), w)
		n += cnt
		if err2 != nil {
			err = err2
			return
		}
	})
	return n, err
}

// Deserialize assigns the params with the values obtained from the reader.
// TODO: use ParamsIterator?
func (m *ParamsSerializer) Deserialize(r io.Reader) (n int, err error) {
	ForEachParam(m, func(param Param) {
		cnt, err2 := mat.UnmarshalBinaryFrom(param.Value(), r)
		n += cnt
		if err2 != nil {
			err = err2
			return
		}
	})
	return n, err
}
