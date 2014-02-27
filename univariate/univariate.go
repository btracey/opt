package univariate

import (
	"math"

	"github.com/btracey/opt/common"
	"github.com/btracey/opt/write"
)

type Objective interface {
	Obj(x float64) float64
}

type Grader interface {
	Grad(x float64) float64
}

type ObjGrader interface {
	ObjGrad(x float64) (f float64, g float64)
}

// Settings is a structure containing settings for univariate
// optimizers. Some settings may not apply to certain algorithms
type Settings struct {
	*common.CommonSettings
	*common.SingleOutputSettings
	InitialObjective float64 // The value of the objective function at the initial location
	InitialGradient  float64 // The value of the gradient at the initial location
}

// DefaultSettings returns the default settings for univariate optimizers.
// The default behavior is to run the optimizer until convergence. If it is desired
// that it end earlier, consider changing MaximumIterations, MaximumFunctionValues,
// and MaximumRuntime
func DefaultSettings() *Settings {
	return &Settings{
		CommonSettings:       common.DefaultCommonSettings(),
		SingleOutputSettings: common.DefaultSingleOutputSettings(),
		InitialObjective:     math.NaN(),
		InitialGradient:      math.NaN(),
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

	// TODO: Maybe think about best, but need a way to deal with floating
	// point issues near the minimum

	objCurr  float64
	gradCurr float64
	locCurr  float64
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
	v = append(v, &write.Value{Heading: "Obj", Value: u.objCurr})
	v = append(v, &write.Value{Heading: "Grad", Value: u.gradCurr})
	return v
}

func (u *Helper) Init(s *Settings, objectiveFunction interface{}, initObj, initGrad float64) {
	u.Common.Init(s.CommonSettings, objectiveFunction)
	u.SingleOutput.Init(s.SingleOutputSettings, initObj, math.Abs(initGrad))

	// TODO: This should get the init loc

	u.objCurr = initObj
	u.gradCurr = initGrad
	u.locCurr = 0
}

func (u *Helper) Iterate(loc, obj, grad float64, nFunEvals int) {
	u.Common.Iterate(nFunEvals)
	u.SingleOutput.Iterate(math.Abs(grad), obj)

	u.locCurr = loc
	u.objCurr = obj
	u.gradCurr = grad
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
		Obj:          u.objCurr,
		Loc:          u.locCurr,
		Grad:         u.gradCurr,
	}
}

type Result struct {
	*common.CommonResult
	Obj  float64 // Lowest found value of the objective function (may not be a minimum from early convergence)
	Loc  float64 // Location where Obj was obtained
	Grad float64 // Gradient where Obj was obtained
}
