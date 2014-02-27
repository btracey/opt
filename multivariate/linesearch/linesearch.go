package linesearch

import (
	"errors"
	"fmt"
	"math"

	"github.com/btracey/opt/common"
	"github.com/btracey/opt/univariate"
	"github.com/gonum/floats"
)

type SetWolfeConditioner interface {
	SetWolfeConditions(fConst, gConst float64, isStrong bool) error
}

type Notconverged struct {
	common.Status
}

func (n Notconverged) Error() string {
	return fmt.Sprintf("Line search terminated not by wolfe conditions. Instead %v", n.Status)
}

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

// Settings contains settings for linesearch
type Settings struct {
	*univariate.Settings
	Wolfe     *WolfeSettings
	Optimizer LinesearchOptimizer
}

type LinesearchOptimizer interface {
	univariate.GradOptimizer
	SetInitStep(float64)
}

func DefaultSettings() *Settings {
	s := &Settings{
		Settings: univariate.DefaultSettings(),
		Wolfe: &WolfeSettings{
			FunConst:  0,
			GradConst: 0.9,
			Type:      Strong,
		},
		//Optimizer: NewDcstep(),
		Optimizer: &univariate.Bisection{},
	}
	s.DisplayWriters = nil
	s.MaximumIterations = 100
	s.GradAbsTol = 0
	return s
}

// TODO: Implement Grad-Free wolfe type May need to change NewWolfeConditions
// to DefaultWolfeConditions(Type)

// TODO: Work on numerical stability

// TODO: Move wolfe to common to avoid import cycle

type WolfeType int

const (
	Strong = iota
	Weak
)

type WolfeSettings struct {
	FunConst  float64
	GradConst float64
	Type      WolfeType
}

type WolfeConditions struct {
	funConst  float64
	gradConst float64
	t         WolfeType

	currObj  float64
	currGrad float64
	initObj  float64
	initGrad float64
	step     float64
}

func (w *WolfeConditions) Init(settings *WolfeSettings, initObj, initGrad float64) {
	w.funConst = settings.FunConst
	w.gradConst = settings.GradConst
	w.t = settings.Type

	w.initObj = initObj
	w.initGrad = initGrad

	w.currObj = initObj
	w.currGrad = initGrad

	if w.t != Strong && w.t != Weak {
		panic("Bad type for wolfe conditions")
	}
	if w.funConst < 0 {
		panic("fun const negative")
	}
	if w.gradConst <= 0 {
		panic("grad const not positive")
	}
}

func (w *WolfeConditions) Iterate(step, obj, grad float64) {
	w.currObj = obj
	w.currGrad = grad
	w.step = step
	if step < 0 {
		panic("WolfeConditions: received a negative step")
	}
}

func (w *WolfeConditions) Status() common.Status {
	switch w.t {
	case Strong:

		/*
			fmt.Println("fun const = ", w.funConst)
			fmt.Println("curr obj = ", w.currObj)
			fmt.Println("init obj = ", w.initObj)
			fmt.Println("fun const", w.funConst)
			fmt.Println("step", w.step)
			fmt.Println("init grad", w.initGrad)
			fmt.Println("comp  = ", w.initObj+w.funConst*w.step*w.initGrad)

			// strictly greater than because when we get close to the minimum, the
			// gradient may change even though the actual function value is the same

			fmt.Println("wolfe const", w.gradConst)
			fmt.Println("wolfe init grad = ", math.Abs(w.initGrad))
			fmt.Println("wolfe abs curr grad = ", math.Abs(w.currGrad))
			fmt.Println("wolfe ratio = ", math.Abs(w.currGrad)/math.Abs(w.initGrad))
		*/
		if w.currObj > w.initObj+w.funConst*w.step*w.initGrad {
			//&& !floats.EqualWithinAbsOrRel(w.currObj, w.initObj, 1e-15, 1e-15) {
			return common.Continue
		}
		if math.Abs(w.currGrad) >= w.gradConst*math.Abs(w.initGrad) {
			return common.Continue
		}
		return common.WolfeConditionsMet
	case Weak:
		if w.currObj >= w.initObj+w.funConst*w.step*w.currGrad {
			return common.Continue
		}
		if w.currGrad <= w.gradConst*w.initGrad {
			return common.Continue
		}
		return common.WolfeConditionsMet
	default:
		panic("wolfe: unknown type")
	}
}

// TODO: Check if can evaluate derivative and function separately to save derivative
// evaluations where appropriate

// LinesearchFun is a type which is the one-dimensional funciton that projects
// the multidimensional function onto the line
type linesearchFun struct {
	fun ObjGrader
	//wolfe WolfeConditions

	searchVector []float64

	//direction   []float64 // unit vector
	initLoc    []float64
	currLoc    []float64
	currLocCpy []float64
	currGrad   []float64
	currObj    float64
	currStep   float64
	//nrmConst float64
}

func (l *linesearchFun) Obj(step float64) float64 {
	f, _ := l.ObjGrad(step)
	return f
}

func (l *linesearchFun) ObjGrad(step float64) (f float64, g float64) {
	l.currStep = step
	// Take the step (need to add back in the scaling)
	for i, val := range l.searchVector {
		l.currLoc[i] = val*step + l.initLoc[i]
	}

	copy(l.currLocCpy, l.currLoc) // In case user function modifies
	f = l.fun.ObjGrad(l.currLocCpy, l.currGrad)

	// Find the gradient in the direction of the search vector
	g = floats.Dot(l.searchVector, l.currGrad)
	l.currObj = f
	/*
		fmt.Println("step = ", step)
		fmt.Println("f = ", f)
		fmt.Println("g = ", g)
	*/
	return f, g
}

// Result is a struct for returning the result from a linesearch
type Result struct {
	Loc       []float64
	Obj       float64
	Grad      []float64
	Step      float64
	NFunEvals int
}

// GradFreeLinesearch performs a gradient-free linesearch on the objective. If
// the strong wolfe conditions are used, fun must be an ObjGrad
func GradLinesearch(settings *Settings,
	fun ObjGrader, searchVector []float64, initLoc []float64, initObj float64, initGrad []float64, prevObj float64) (*Result, error) {

	if len(searchVector) != len(initLoc) {
		return nil, errors.New("linesearch: search vector length does not match init loc")
	}

	// TODO: Fix this to remove allocations
	// TODO: Add error checking
	wrapper := univariate.NewGradWrapper(settings.Optimizer)

	line := &linesearchFun{
		fun:          fun,
		searchVector: searchVector,
		initLoc:      initLoc,
		currLoc:      make([]float64, len(initLoc)),
		currLocCpy:   make([]float64, len(initLoc)),
		currGrad:     make([]float64, len(initLoc)),
	}

	wolfe := &WolfeConditions{}

	grad := floats.Dot(searchVector, initGrad)

	/*

		nrmInitGrad := floats.Norm(initGrad, 2)
		nrmSearchVector := floats.Norm(searchVector, 2)
		//	fmt.Println("grad nrm", floats.Norm(initGrad, 2))

		costheta := math.Abs(grad) / nrmInitGrad
		costheta /= nrmSearchVector

		// Don't let us walk too far away from the gradient

		// TODO: What should this actually be
		if costheta < 0.3 {
			for i := range searchVector {
				searchVector[i] = -initGrad[i] / nrmInitGrad * nrmSearchVector
			}
			grad = floats.Dot(searchVector, initGrad)
		}
	*/

	//nrmSearchVector := floats.Norm(searchVector, 2)

	//floats.Scale(1/nrmSearchVector, searchVector)
	//defer floats.Scale(nrmSearchVector, searchVector)

	nrmConst := 1.0
	dirGrad := grad / nrmConst
	if grad > 0 {
		return nil, errors.New("Initial gradient is positive")
	}
	wolfe.Init(settings.Wolfe, initObj, dirGrad)

	settings.InitialGradient = grad
	settings.InitialObjective = initObj

	// alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)

	//fmt.Println("initObj", initObj, "prevObj", prevObj)
	//fmt.Println("grad", grad)
	initStepSize := 1.01 * 2 * (initObj - prevObj) / (grad) // Got this trick from scipy
	//fmt.Println("initstepsize", initStepSize)
	initStepSize = math.Min(1.0, initStepSize)
	//fmt.Println("initstepsize", initStepSize)
	if initStepSize <= 0 {
		fmt.Println("init size negative")
		initStepSize = 1.0
	}
	//initStepSize *= nrmSearchVector

	settings.Optimizer.SetInitStep(initStepSize)

	wolfeConditioner, ok := settings.Optimizer.(SetWolfeConditioner)
	if ok {
		err := wolfeConditioner.SetWolfeConditions(settings.Wolfe.FunConst, settings.Wolfe.GradConst, settings.Wolfe.Type == Strong)
		if err != nil {
			return nil, err
		}
	}

	err := wrapper.Init(settings.Settings, line, 0)
	if err != nil {
		panic("error initializing: " + err.Error())
	}

	/*
		loss1 := line.Obj(0 + 1e-6)
		loss2 := line.Obj(0 - 1e-6)
		_, deriv := line.ObjGrad(0)
	*/
	/*
		fmt.Println("Initial grad", grad)
		fmt.Println("true deriv", deriv)
		fmt.Println("fd deriv", (loss1-loss2)/(2e-6))
	*/
	// Perform an augmented optimization where the wolfe conditions are checked
	// at every step
	var status common.Status
	//var prevObj float64
	//var prevStep float64
	//fmt.Println()
	//fmt.Println("init search norm is ", nrmSearchVector)
	//fmt.Println("init dir grad is ", dirGrad)
	//fmt.Println("real grad norm is ", floats.Norm(initGrad, 2))
	//fmt.Println("init grad is ", dirGrad)
	//fmt.Println("new iteration")
	for {
		status = wolfe.Status()
		if status != common.Continue {
			break
		}
		status = wrapper.Status()
		if status != common.Continue {
			break
		}

		step, obj, _, err := wrapper.Iterate()
		if err != nil {
			return nil, err
		}

		grad := line.currGrad
		dirGrad := floats.Dot(searchVector, grad) / nrmConst

		wolfe.Iterate(step, obj, dirGrad)
	}

	lineresult := wrapper.Result(status)

	result := &Result{
		Loc:       line.currLoc,
		Obj:       line.currObj,
		Grad:      line.currGrad,
		Step:      line.currStep,
		NFunEvals: lineresult.FunctionEvaluations,
	}

	//	fmt.Println("final loc", line.currLoc)
	if status != common.WolfeConditionsMet {
		return result, Notconverged{status}
	}
	return result, nil
}
