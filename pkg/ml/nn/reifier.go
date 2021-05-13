// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"reflect"
)

type reifier struct {
	g    *ag.Graph
	mode ProcessingMode
}

func newReifier(g *ag.Graph, mode ProcessingMode) *reifier {
	return &reifier{
		g:    g,
		mode: mode,
	}
}

func (r *reifier) reify(m Model) Model {
	p := r.reifyStruct(m).(Model)
	p.InitProcessor()
	return p
}

func (r *reifier) reifyStruct(rawSource interface{}) interface{} {
	source := reflect.ValueOf(rawSource)
	sourceType := reflect.TypeOf(rawSource)

	sourceIsPointer := source.Kind() == reflect.Ptr
	if sourceIsPointer {
		sourceIsPointer = true
		source = source.Elem()
		sourceType = sourceType.Elem()
	}

	destPointer := reflect.New(sourceType)
	dest := destPointer.Elem()

	numFields := source.NumField()
	for fieldIndex := 0; fieldIndex < numFields; fieldIndex++ {
		sourceField, destField := source.Field(fieldIndex), dest.Field(fieldIndex)
		tag, err := parseModuleFieldTag(sourceType.Field(fieldIndex).Tag.Get("spago"))
		if err != nil {
			panic(err)
		}

		if tag.Scope == processorModuleFieldScope || !sourceField.CanInterface() {
			continue // skip any initialization
		}

		if tag.Scope == modelModuleFieldScope {
			destField.Set(sourceField)
			continue
		}

		r.reifyStructField(sourceField, destField, tag)
	}

	if !sourceIsPointer {
		return dest.Interface()
	}
	return destPointer.Interface()
}

func (r *reifier) reifyStructField(sourceField, destField reflect.Value, tag moduleFieldTag) {
	switch sourceFieldT := sourceField.Interface().(type) {
	case *ag.Graph:
		destField.Set(reflect.ValueOf(r.g))
	case ProcessingMode:
		destField.Set(reflect.ValueOf(r.mode))
	case BaseModel, *BaseModel:
		destField.Set(reflect.ValueOf(r.reifyStruct(sourceFieldT)))
	case Param:
		destField.Set(reflect.ValueOf(r.reifyParam(sourceFieldT.(*param))))
	case []Param:
		destField.Set(reflect.ValueOf(r.reifyParamSlice(sourceFieldT)))
	case Model:
		destField.Set(reflect.ValueOf(r.reifyModel(sourceFieldT)))
	case []Model:
		destField.Set(reflect.ValueOf(r.reifyModelSlice(sourceFieldT)))
	default:
		switch sourceField.Kind() {
		case reflect.Slice:
			destField.Set(r.reifySlice(sourceField, tag))
		case reflect.Map:
			destField.Set(r.reifyMap(sourceField, tag))
		case reflect.Struct, reflect.Ptr:
			if sourceField.Kind() == reflect.Ptr && sourceField.IsNil() {
				return
			}
			if tag.Type == paramsModuleFieldType {
				destField.Set(reflect.ValueOf(r.reifyStruct(sourceFieldT)))
			} else {
				destField.Set(sourceField)
			}
		default:
			destField.Set(sourceField)
		}
	}
}

func (r *reifier) reifyModel(sourceField Model) Model {
	if isNil(sourceField) {
		return sourceField
	}
	p := Reify(sourceField, r.g, r.mode)
	p.InitProcessor()
	return p
}

func (r *reifier) reifyModelSlice(sourceField []Model) []Model {
	result := make([]Model, len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		result[i] = r.reifyModel(sourceField[i])
	}
	return result
}

func (r *reifier) reifyParam(sourceField *param) Param {
	return sourceField.wrappedParam(r.g)
}

func (r *reifier) reifyParamSlice(sourceField []Param) []Param {
	result := make([]Param, len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		result[i] = r.reifyParam(sourceField[i].(*param))
	}
	return result
}

func (r *reifier) reifySlice(sourceField reflect.Value, tag moduleFieldTag) reflect.Value {
	length := sourceField.Len()
	result := reflect.MakeSlice(sourceField.Type(), length, length)
	isParamsTag := tag.Type == paramsModuleFieldType

	for i := 0; i < length; i++ {
		sourceItem := sourceField.Index(i)
		switch sourceItem.Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			_, isModule := sourceItem.Interface().(Model)
			if isParamsTag || isModule {
				result.Index(i).Set(reflect.ValueOf(r.reifyStruct(sourceItem.Interface())))
			} else {
				return sourceField
			}
		default:
			if isParamsTag {
				panic(`nn: "params"-tagged slice contains items with unexpected type`)
			}
			return sourceField
		}
	}
	return result
}

var paramInterfaceName = reflect.TypeOf((*Param)(nil)).Elem().Name()

func (r *reifier) reifyMap(sourceValue reflect.Value, tag moduleFieldTag) reflect.Value {
	sourceType := reflect.TypeOf(sourceValue.Interface())
	mapValueType := sourceType.Elem()

	if mapValueType.Name() != paramInterfaceName && tag.Type != paramsModuleFieldType {
		return sourceValue
	}

	result := reflect.MakeMap(sourceType)
	mapValueKind := mapValueType.Kind()
	mapRange := sourceValue.MapRange()

	for mapRange.Next() {
		key := mapRange.Key()
		sourceValue := mapRange.Value()

		var destValue reflect.Value
		if p, isParam := sourceValue.Interface().(*param); isParam {
			destValue = reflect.ValueOf(r.reifyParam(p))
		} else if mapValueKind == reflect.Struct || mapValueKind == reflect.Ptr {
			destValue = reflect.ValueOf(r.reifyStruct(sourceValue.Interface()))
		} else {
			panic(`nn: "params"-tagged map contains values with unexpected type`)
		}
		result.SetMapIndex(key, destValue)
	}

	return result
}

func isNil(a interface{}) bool {
	return a == nil || (reflect.ValueOf(a).Kind() == reflect.Ptr && reflect.ValueOf(a).IsNil())
}
