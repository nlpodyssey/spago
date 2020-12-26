// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"reflect"
)

type modelContextualizer struct {
	ctx   Context
	graph *ag.Graph
}

// newModelContextualizer returns a new modelContextualizer.
func newModelContextualizer(ctx Context) modelContextualizer {
	return modelContextualizer{
		ctx:   ctx,
		graph: ctx.Graph,
	}
}

func (mc modelContextualizer) contextualize(m Model) Processor {
	p := mc.contextualizeStruct(m).(Processor)
	p.InitProc()
	return p
}

func (mc modelContextualizer) contextualizeStruct(rawSource interface{}) interface{} {
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

		if tag.Scope == processorModuleFieldScope {
			continue // skip any initialization
		}

		if !sourceField.CanInterface() {
			continue
		}

		switch sourceFieldT := sourceField.Interface().(type) {
		case Context:
			destField.Set(reflect.ValueOf(mc.ctx))
		case BaseModel, *BaseModel:
			destField.Set(reflect.ValueOf(mc.contextualizeStruct(sourceFieldT)))
		case Param:
			destField.Set(reflect.ValueOf(mc.contextualizeParam(sourceFieldT.(*param))))
		case []Param:
			destField.Set(reflect.ValueOf(mc.contextualizeParamSlice(sourceFieldT)))
		case Module:
			destField.Set(reflect.ValueOf(mc.contextualizeModel(sourceFieldT)))
		case []Module:
			destField.Set(reflect.ValueOf(mc.contextualizeModelSlice(sourceFieldT)))
		default:
			switch sourceField.Kind() {
			case reflect.Slice:
				destField.Set(mc.contextualizeSlice(sourceField, tag))
			case reflect.Map:
				destField.Set(mc.contextualizeMap(sourceField, tag))
			case reflect.Struct, reflect.Ptr:
				if sourceField.Kind() == reflect.Ptr && sourceField.IsNil() {
					continue
				}
				if tag.Type == paramsModuleFieldType {
					destField.Set(reflect.ValueOf(mc.contextualizeStruct(sourceFieldT)))
				} else {
					destField.Set(sourceField)
				}
			default:
				destField.Set(sourceField)
			}
		}
	}

	if !sourceIsPointer {
		return dest.Interface()
	}
	return destPointer.Interface()
}

func (mc modelContextualizer) contextualizeModel(sourceField Module) Module {
	p := NewProc(mc.ctx, sourceField)
	p.InitProc()
	return p
}

func (mc modelContextualizer) contextualizeModelSlice(sourceField []Module) []Module {
	result := make([]Module, len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		result[i] = mc.contextualizeModel(sourceField[i])
	}
	return result
}

func (mc modelContextualizer) contextualizeParam(sourceField *param) Param {
	return sourceField.wrappedParam(mc.graph)
}

func (mc modelContextualizer) contextualizeParamSlice(sourceField []Param) []Param {
	result := make([]Param, len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		result[i] = mc.contextualizeParam(sourceField[i].(*param))
	}
	return result
}

func (mc modelContextualizer) contextualizeSlice(sourceField reflect.Value, tag moduleFieldTag) reflect.Value {
	length := sourceField.Len()
	result := reflect.MakeSlice(sourceField.Type(), length, length)

	for i := 0; i < length; i++ {
		sourceItem := sourceField.Index(i)

		switch sourceItem.Kind() {
		case reflect.Struct, reflect.Ptr:
			isParams := tag.Type == paramsModuleFieldType
			_, isModule := sourceItem.Interface().(Module)

			if isParams || isModule {
				result.Index(i).Set(reflect.ValueOf(mc.contextualizeStruct(sourceItem.Interface())))
			} else {
				return sourceField
			}
		default:
			panic(`nn: "params"-tagged slice contains items with unexpected type`)
		}
	}

	return result
}

var paramInterfaceName = reflect.TypeOf((*Param)(nil)).Elem().Name()

func (mc modelContextualizer) contextualizeMap(sourceValue reflect.Value, tag moduleFieldTag) reflect.Value {
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
			destValue = reflect.ValueOf(mc.contextualizeParam(p))
		} else if mapValueKind == reflect.Struct || mapValueKind == reflect.Ptr {
			destValue = reflect.ValueOf(mc.contextualizeStruct(sourceValue.Interface()))
		} else {
			panic(`nn: "params"-tagged map contains values with unexpected type`)
		}
		result.SetMapIndex(key, destValue)
	}

	return result
}
