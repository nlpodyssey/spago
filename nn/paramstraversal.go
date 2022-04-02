// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"fmt"
	"reflect"
	"strings"
	"sync"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/utils"
)

// ParamsTraversalFunc is the function called for each visited Param from
// traversal routines (see for example ForEachParam and ForEachParamStrict).
//
// The arguments are:
//   - param: the value of the visited parameter
//   - name: a suggested meaningful name for the parameter, if available,
//     usually corresponding to the name of a struct's field
//   - pType: type of the parameter, if available, usually derived from
//     a `spago:"type:..."` tag from a struct's field
type ParamsTraversalFunc[T mat.DType] func(param Param[T], name string, pType ParamsType)

// ParamsTraverser allows you to define a custom procedure to traverse the parameters of a model.
// If a model implements this procedure, it will take precedence over the regular parameters visit.
type ParamsTraverser[T mat.DType] interface {
	// TraverseParams calls ParamsTraversalFunc for each visited Param.
	TraverseParams(callback ParamsTraversalFunc[T])
}

// paramsTraversal allows the traversal of Model parameters.
// The given callback is invoked for each parameter of the Model.
// If exploreSubModels is true, every nested Model and its parameters are
// also visited.
type paramsTraversal[T mat.DType] struct {
	callback         ParamsTraversalFunc[T]
	exploreSubModels bool
}

// newParamsTraversal returns a new paramsTraversal.
func newParamsTraversal[T mat.DType](fn ParamsTraversalFunc[T], exploreSubModels bool) paramsTraversal[T] {
	return paramsTraversal[T]{
		callback:         fn,
		exploreSubModels: exploreSubModels,
	}
}

// walk iterates through all the parameters of m.
// TODO: don't loop the field every time, use a lazy initialized "params list" instead
func (pt paramsTraversal[_]) walk(m any) {
	utils.ForEachField(m, func(field any, name string, rTag reflect.StructTag) {
		tag, err := parseModuleFieldTag(rTag.Get("spago"))
		if err != nil {
			panic(err)
		}
		v := reflect.ValueOf(field)
		switch v.Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			pt.walkStructOrPtr(field, name, tag)
		case reflect.Slice:
			pt.walkSlice(v, name, tag)
		case reflect.Map:
			pt.walkMap(v, name, tag)
		}
	})
}

func (pt paramsTraversal[T]) walkStructOrPtr(item any, name string, tag moduleFieldTag) {
	v := reflect.ValueOf(item)
	if v.Kind() == reflect.Ptr && v.Elem().Kind() != reflect.Struct {
		return
	}
	switch itemT := item.(type) {
	case Param[T]:
		pt.callback(itemT, name, tag.paramType())
	case ParamsTraverser[T]:
		itemT.TraverseParams(pt.callback)
	case Model[T]:
		if pt.exploreSubModels {
			pt.walk(item)
		}
	case ag.Differentiable[T]:
		_, isModel := itemT.(Model[T])
		if !isModel {
			pt.walk(item)
		}
	case *sync.Map:
		pt.walkSyncMap(itemT, name, tag)
	default:
		return
	}
}

func (pt paramsTraversal[_]) walkSyncMap(i *sync.Map, name string, tag moduleFieldTag) {
	i.Range(func(key, value any) bool {
		switch k := key.(type) {
		case string:
			key = k
		case int:
			key = fmt.Sprintf("%d", k)
		default:
			return false // skip map if the key is not a string or an int
		}

		name := strings.ToLower(fmt.Sprintf("%s.%s", name, key))
		switch reflect.ValueOf(value).Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			pt.walkStructOrPtr(value, name, tag)
		default:
			return false // skip
		}
		return true
	})
}

func (pt paramsTraversal[_]) walkSlice(v reflect.Value, name string, tag moduleFieldTag) {
	length := v.Len()
	for i := 0; i < length; i++ {
		p := v.Index(i)
		switch p.Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			pt.walkStructOrPtr(p.Interface(), name, tag)
		default:
			return // skip
		}
	}
}

func (pt paramsTraversal[_]) walkMap(v reflect.Value, name string, tag moduleFieldTag) {
	mapRange := v.MapRange()
	for mapRange.Next() {
		key := ""
		switch k := mapRange.Key().Interface().(type) {
		case string:
			key = k
		case int:
			key = fmt.Sprintf("%d", k)
		default:
			return // skip map if the key is not a string or an int
		}

		name := strings.ToLower(fmt.Sprintf("%s.%s", name, key))
		switch mapRange.Value().Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			pt.walkStructOrPtr(mapRange.Value().Interface(), name, tag)
		default:
			return // skip
		}
	}
}
