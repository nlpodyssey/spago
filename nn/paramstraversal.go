// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"fmt"
	"reflect"
	"sync"
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
type ParamsTraversalFunc func(param Param, name string, pType ParamsType)

// ModelsTraversalFunc is the function called for each visited Model from
// traversal routines (see for example ForEachModels).
//
// The arguments are:
//   - model: the value of the visited model
//   - name: a suggested meaningful name for the model, if available,
//     usually corresponding to the name of a struct's field
type ModelsTraversalFunc func(model Model, name string)

// ParamsTraverser allows you to define a custom procedure to traverse the parameters of a model.
// If a model implements this procedure, it will take precedence over the regular parameters visit.
type ParamsTraverser interface {
	// TraverseParams calls ParamsTraversalFunc for each visited Param.
	TraverseParams(callback ParamsTraversalFunc)
}

// paramsTraversal allows the traversal of Model parameters.
// The given paramsFunc is invoked for each parameter of the Model.
// If exploreSubModels is true, every nested Model and its parameters are
// also visited.
type paramsTraversal struct {
	paramsFunc       ParamsTraversalFunc
	modelsFunc       ModelsTraversalFunc
	exploreSubModels bool
}

// ForEachParam iterate all the parameters of a model also exploring the sub-parameters recursively.
func ForEachParam(m Model, fn ParamsTraversalFunc) {
	paramsTraversal{
		paramsFunc:       fn,
		modelsFunc:       nil,
		exploreSubModels: true,
	}.walk(m)
}

// ForEachParamStrict iterate all the parameters of a model without exploring the sub-models.
func ForEachParamStrict(m Model, fn ParamsTraversalFunc) {
	paramsTraversal{
		paramsFunc:       fn,
		modelsFunc:       nil,
		exploreSubModels: false,
	}.walk(m)
}

// ForEachModel iterate all the sub-models of a model.
func ForEachModel(m Model, fn ModelsTraversalFunc) {
	fn(m, "")
	paramsTraversal{
		paramsFunc:       nil,
		modelsFunc:       fn,
		exploreSubModels: true,
	}.walk(m)
}

// walk iterates through all the parameters of m.
func (pt paramsTraversal) walk(m any) {
	if m, ok := m.(ParamsTraverser); ok {
		m.TraverseParams(pt.paramsFunc)
		return
	}
	forEachField(m, func(field any, name string, rTag reflect.StructTag) {
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

func (pt paramsTraversal) walkStructOrPtr(item any, name string, tag moduleFieldTag) bool {
	v := reflect.ValueOf(item)
	if v.Kind() == reflect.Ptr && v.Elem().Kind() != reflect.Struct {
		return false
	}
	switch itemT := item.(type) {
	case Module, *Module:
		// skip
	case Param:
		if pt.paramsFunc != nil {
			pt.paramsFunc(itemT, name, tag.paramType())
		}
	case ParamsTraverser:
		if pt.paramsFunc != nil {
			itemT.TraverseParams(pt.paramsFunc)
		}
	case Model:
		if pt.exploreSubModels {
			if pt.modelsFunc != nil {
				pt.modelsFunc(itemT, name)
			}
			pt.walk(item)
		}
	case *sync.Map:
		pt.walkSyncMap(itemT, name, tag)
	default:
		return false
	}
	return true
}

func (pt paramsTraversal) walkSyncMap(i *sync.Map, name string, tag moduleFieldTag) {
	i.Range(func(key, value any) bool {
		switch k := key.(type) {
		case string:
			key = k
		case int:
			key = fmt.Sprintf("%d", k)
		default:
			return false // skip map if the key is not a string or an int
		}

		name := fmt.Sprintf("%s.%s", name, key)
		switch reflect.ValueOf(value).Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			if !pt.walkStructOrPtr(value, name, tag) {
				return false
			}
		default:
			return false // skip
		}
		return true
	})
}

func (pt paramsTraversal) walkSlice(v reflect.Value, name string, tag moduleFieldTag) {
	length := v.Len()
	for i := 0; i < length; i++ {
		p := v.Index(i)
		switch p.Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			if !pt.walkStructOrPtr(p.Interface(), name, tag) {
				return
			}
		default:
			return // skip
		}
	}
}

func (pt paramsTraversal) walkMap(v reflect.Value, name string, tag moduleFieldTag) {
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

		name := fmt.Sprintf("%s.%s", name, key)
		switch mapRange.Value().Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			if !pt.walkStructOrPtr(mapRange.Value().Interface(), name, tag) {
				return
			}
		default:
			return // skip
		}
	}
}

// forEachField calls the paramsFunc for each field of the struct i.
func forEachField(i any, callback func(field any, name string, tag reflect.StructTag)) {
	v := reflect.ValueOf(i)
	t := reflect.TypeOf(i)

	if v.Kind() == reflect.Ptr {
		v = v.Elem()
		t = t.Elem()
	}

	length := v.NumField()
	for i := 0; i < length; i++ {
		vField := v.Field(i)
		tField := t.Field(i)
		if vField.CanInterface() {
			if vField.Kind() == reflect.Ptr && vField.IsNil() {
				continue
			}
			callback(vField.Interface(), tField.Name, tField.Tag)
		}
	}
}
