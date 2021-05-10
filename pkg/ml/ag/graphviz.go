// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"github.com/awalterschulze/gographviz"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"reflect"
)

// GraphvizGraph creates a gographviz graph representation of the Graph.
func (g *Graph) GraphvizGraph() (gographviz.Interface, error) {
	gg := gographviz.NewEscape()

	if err := gg.SetDir(true); err != nil {
		return nil, err
	}
	if err := gg.AddAttr("", "rankdir", "LR"); err != nil {
		return nil, err
	}

	for _, node := range g.nodes {
		switch nt := node.(type) {
		case *variable:
			if err := g.addGVVariable(gg, nt); err != nil {
				return nil, err
			}
		case *operator:
			if err := g.addGVOperator(gg, nt); err != nil {
				return nil, err
			}
		// TODO: case *param
		// TODO: case *wrappedParam
		// TODO: case *wrapper
		default:
			return nil, fmt.Errorf("unexpected node type %T", node)
		}
	}
	return gg, nil
}

func (g *Graph) addGVVariable(gg gographviz.Interface, v *variable) error {
	id := fmt.Sprintf("%d", v.ID())
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			variable<BR />
			%s
		>`,
		v.ID(),
		gvMatrixShape(v.value),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return gg.AddNode("", id, attrs)
}

func (g *Graph) addGVOperator(gg gographviz.Interface, op *operator) error {
	operatorID := fmt.Sprintf("%d", op.ID())
	funcName := reflect.ValueOf(op.function).Elem().Type().Name()
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			<B>%s</B><BR />
			%s
		>`,
		op.ID(),
		funcName,
		gvMatrixShape(op.Value()),
	)
	attrs := map[string]string{
		"label": label,
	}
	if err := gg.AddNode("", operatorID, attrs); err != nil {
		return err
	}

	for _, operand := range op.operands {
		operandID := fmt.Sprintf("%d", operand.ID())
		if err := gg.AddEdge(operandID, operatorID, true, nil); err != nil {
			return err
		}
	}
	return nil
}

func gvMatrixShape(m mat.Matrix) string {
	if m == nil {
		return ""
	}
	if m.IsScalar() {
		return "scalar"
	}
	return fmt.Sprintf("%d × %d", m.Rows(), m.Columns())
}
