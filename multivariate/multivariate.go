package multivariate

import (
	"math"

	"github.com/btracey/opt/common"
	"github.com/btracey/opt/write"
	"github.com/gonum/floats"
)

type Objective interface {
	Obj(x []float64) float64
}

type Grader interface {
	Grad(x []float64, g []float64)
}

// puts gradient in place
type ObjGrader interface {
	ObjGrad(x []float64, g []float64) (f float64)
}

// Settings is a structure containing settings for multivariate
// optimizers. Some settings may not apply to certain algorithms
type Settings struct {
	*common.CommonSettings
	*common.SingleOutputSettings
	InitialObjective float64
	InitialGradient  []float64
}

// DefaultSettings returns the default settings for multivariate optimizers.
// The default behavior is to run the optimizer until convergence. If it is desired
// that it end earlier, consider changing MaximumIterations, MaximumFunctionValues,
// and MaximumRuntime
func DefaultSettings() *Settings {
	return &Settings{
		CommonSettings:       common.DefaultCommonSettings(),
		SingleOutputSettings: common.DefaultSingleOutputSettings(),
		InitialObjective:     math.NaN(),
		InitialGradient:      nil,
	}
}

// Helper is a helper struct for optimizers. Not intended for use by
// callers of optimization functions, but exported to aid others who are building
// optimization algorithms
//
// Optimization implementers should call Init() at the beginning of an optimization run
// and should call Status() to check tolerances. At the end of every interation should call
// Iterate()
type Helper struct {
	*common.Common
	*common.SingleOutput

	objBest     float64
	gradBest    []float64
	locBest     []float64
	gradNrmBest float64
}

// NewHelper creates a new univariate type and adds itself to the data adders
func NewHelper() *Helper {
	u := &Helper{
		Common:       common.NewCommon(),
		SingleOutput: common.NewSingleOutput(),
	}
	u.AddDataAdder(u)
	return u
}

func (u *Helper) AppendWriteData(v []*write.Value) []*write.Value {
	v = append(v, &write.Value{Heading: "Obj", Value: u.objBest})
	v = append(v, &write.Value{Heading: "Grad", Value: u.gradNrmBest})
	return v
}

func (u *Helper) Init(s *Settings, objectiveFunction interface{}, initObj float64, initGrad []float64) {
	u.Common.Init(s.CommonSettings, objectiveFunction)

	gradNrm := floats.Norm(initGrad, 2)

	u.SingleOutput.Init(s.SingleOutputSettings, initObj, gradNrm)

	u.objBest = math.Inf(1)
	u.gradBest = nil
	u.locBest = nil
	u.gradNrmBest = gradNrm
}

func (u *Helper) Iterate(loc []float64, obj float64, grad []float64, nFunEvals int) {
	u.Common.Iterate(nFunEvals)
	gradNrm := floats.Norm(grad, 2)
	/*
		fmt.Println("multivariate loc = ", loc)
		fmt.Println("multivariate, obj = ", obj)
		fmt.Println("multivariate, grad = ", grad)
		fmt.Println("multivariate, gradNrm ", gradNrm)
	*/

	u.SingleOutput.Iterate(gradNrm, obj)

	if obj <= u.objBest {
		u.objBest = obj
		u.gradBest = grad
		u.locBest = loc
		u.gradNrmBest = gradNrm
	}
}

func (u *Helper) Status() common.Status {
	status := u.SingleOutput.Status()
	if status != common.Continue {
		return status
	}
	status = u.Common.Status()
	if status != common.Continue {
		return status
	}
	return common.Continue
}

func (u *Helper) Result(status common.Status) *Result {
	return &Result{
		CommonResult: u.Common.Result(status),
		Obj:          u.objBest,
		Loc:          u.locBest,
		Grad:         u.gradBest,
	}
}

type Result struct {
	*common.CommonResult
	Obj  float64   // Lowest found value of the objective function (may not be a minimum from early convergence)
	Loc  []float64 // Location where Obj was obtained
	Grad []float64 // Gradient where Obj was obtained
}
