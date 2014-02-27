package multivariate

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/btracey/opt/common"

	"github.com/gonum/blas/cblas"
	"github.com/gonum/blas/dbw"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

func init() {
	dbw.Register(cblas.Blas{})
	mat64.Register(cblas.Blas{})
}

const MISO_TOLERANCE = 1e-12

type Bowl struct {
	A *mat64.Dense
	B *mat64.Dense
	m int
	n int
}

func NewBowl(m, n int) *Bowl {
	b := &Bowl{
		A: mat64.NewDense(m, n, nil),
		B: mat64.NewDense(m, 1, nil),
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			b.A.Set(i, j, float64(i*m+j*j)/100)
		}
		b.B.Set(i, 0, float64(i*i))
	}
	b.m = m
	b.n = n
	return b
}

func (b *Bowl) ObjGrad(x []float64, grad []float64) (loss float64) {
	//fmt.Println("x = ", x)
	xmat := mat64.NewDense(len(x), 1, x)
	newMat := mat64.NewDense(b.m, 1, nil)

	//fmt.Println(xmat)
	//fmt.Println(...)
	newMat.Mul(b.A, xmat)

	/*
		fmt.Println("A", b.A)
		fmt.Println("B", b.B)
		fmt.Println(xmat)
		fmt.Println(newMat)
	*/
	//fmt.Println("newmat", newMat)
	for i := range grad {
		grad[i] = 0
	}

	for i := 0; i < b.m; i++ {
		diff := newMat.At(i, 0) - b.B.At(i, 0)
		//fmt.Println("diff")
		loss += diff * diff
		for j := range grad {
			grad[j] += 2 * diff * b.A.At(i, j)
		}
	}
	//fmt.Println("grad", grad)
	//fmt.Println("loss", loss)
	//panic("blah")
	return loss
}

func (b *Bowl) OptLoc() []float64 {
	return mat64.Solve(b.A, b.B).RawMatrix().Data
}

func (b *Bowl) OptVal() float64 {
	return 0
}

type Rosenbrock struct {
	nDim int
}

func (r *Rosenbrock) ObjGrad(x []float64, deriv []float64) (sum float64) {
	sum = 0

	for i := range deriv {
		deriv[i] = 0
	}

	for i := 0; i < len(x)-1; i++ {
		sum += math.Pow(1-x[i], 2) + 100*math.Pow(x[i+1]-math.Pow(x[i], 2), 2)
	}
	for i := 0; i < len(x)-1; i++ {
		deriv[i] += -1 * 2 * (1 - x[i])
		deriv[i] += 2 * 100 * (x[i+1] - math.Pow(x[i], 2)) * (-2 * x[i])
	}
	for i := 1; i < len(x); i++ {
		deriv[i] += 2 * 100 * (x[i] - math.Pow(x[i-1], 2))
	}
	return sum
}

func (r *Rosenbrock) OptVal() float64 {
	return 0
}

func (r *Rosenbrock) OptLoc() []float64 {
	ans := make([]float64, r.nDim)
	floats.AddConst(1, ans)
	return ans
}

type GradTestFunction interface {
	ObjGrader
	OptVal() float64
	OptLoc() []float64
}

type GradTest struct {
	GradTestFunction
	InitLoc []float64
	name    string
}

func RandRosen(nDim int, low, high float64) GradTest {
	rosen := GradTest{
		GradTestFunction: &Rosenbrock{nDim: nDim},
		name:             "TenDRosenbrock rand, nDim = " + strconv.Itoa(nDim),
		InitLoc:          make([]float64, nDim),
	}
	for i := range rosen.InitLoc {
		rosen.InitLoc[i] = rand.Float64()*(high-low) + low
	}
	return rosen
}

func GradFunctions() []GradTest {
	misotest := make([]GradTest, 0)

	misotest = append(misotest, GradTest{
		GradTestFunction: &Rosenbrock{5},
		InitLoc:          []float64{1.3, 0.7, 0.8, 1.9, 1.2},
		name:             "rosen",
	},
	)

	misotest = append(misotest, RandRosen(10, -10, 10))
	misotest = append(misotest, RandRosen(50, -2, 2))
	misotest = append(misotest, RandRosen(50, -100, 100))

	misotest = append(misotest, GradTest{NewBowl(10, 3), []float64{2, 2, 2}, "bowl"})

	return misotest
}

func GradBasedTest(t *testing.T, opter GradOptimizer) {
	funcs := GradFunctions()
	for _, fun := range funcs {
		// Run it once until very converged

		fmt.Println(fun.name)

		settings := DefaultSettings()
		settings.GradAbsTol = MISO_TOLERANCE
		settings.MaximumFunctionEvaluations = 1000

		//optVal, optLoc, c, err := opter.Optimize(fun, fun.InitLoc)
		result, err := OptimizeGrad(fun, fun.InitLoc, settings, opter)
		if err != nil {
			t.Errorf("Error optimizing: %v", err)
			continue
		}
		fmt.Println("n iter", result.Iterations)
		fmt.Println("n fun evals", result.FunctionEvaluations)

		c := result.Status

		if c == common.Continue {
			t.Errorf("Finished optimizing without error and status is continue")
			return
		}
		if c != common.GradAbsTol {
			t.Errorf("For function " + fun.name + " status is not GradAbsTol.")
			return
		}

		/*
			firstObjVal := optVal
			if math.Abs(firstObjVal-fun.OptVal()) > MISO_TOLERANCE {
				t.Errorf("For function "+fun.name+" optimum value not found. %v found, %v expected", firstObjVal, fun.OptVal())
				return
			}
			firstLocVal := optLoc
			if !floats.EqualApprox(firstLocVal, fun.OptLoc(), MISO_TOLERANCE) {
				t.Errorf("For function "+fun.name+" optimum location not found. %v found, %v expected", firstLocVal, fun.OptLoc())
			}
		*/

		firstNFunEvals := result.FunctionEvaluations
		firstNIterations := result.Iterations
		// Hack to reset FunEvals

		// Run it again to test that the reset works fine
		//opter.Loc().SetInit(fun.InitLoc)
		//_, _, c, err = opter.Optimize(fun, fun.InitLoc)
		result, err = OptimizeGrad(fun, fun.InitLoc, settings, opter)
		if err != nil {
			t.Errorf("Error while re-using optimizer: ", err.Error())
		}

		if result.FunctionEvaluations != firstNFunEvals {
			t.Errorf("For function " + fun.name + "Different number of fun evals second time")
		}

		if result.Status != common.GradAbsTol {
			t.Errorf("For function " + fun.name + " status is not GradAbsTol second time")
		}

		if result.Iterations != firstNIterations {
			fmt.Println(result.Iterations, firstNIterations)
			t.Errorf("For function " + fun.name + "Different number of fun evals second time")
		}

	}
}
