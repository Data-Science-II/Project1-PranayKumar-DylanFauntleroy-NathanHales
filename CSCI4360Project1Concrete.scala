//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dylan Fauntleroy, Nathan Hales, Pranay Kumar, and some repeated code from John Miller
 *  @version 1.6
 *  @date    2/24/2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Project1 Code
 */

package scalation.analytics

import scala.collection.mutable.Set
import scala.util.control.Breaks.{break, breakable}

import scalation.linalgebra._
import scalation.math.noDouble
import scalation.plot.{Plot, PlotM}
import scalation.random.CDF.studentTCDF
import scalation.stat.Statistic
import scalation.stat.StatVector.corr
import scalation.util.banner
import scalation.util.Unicode.sub

import Fit._
import RegTechnique._

//RegTest16 Tests forward selection with MLR for concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest16` object tests the `Regression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest16
 */
object RegressionTest16 extends App
{
    //Concrete code for MLR
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "Concrete_Data.csv", "Concrete", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val rg = new Regression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)

    val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
    val k = cols.size
	println("k:")
	println(k)
	
    println (s"k = $k, n = ${x.dim2}")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Regression", lines = true)

    println (s"rSq = $rSq")


} // RegressionTest16 object

//RegTest17 Tests backward elimination with MLR for concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest17` object tests the `Regression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest17
 */
object RegressionTest17 extends App
{
    //Concrete code for MLR
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "Concrete_Data.csv", "Concrete", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val rg = new Regression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)

    val (cols, rSq) = rg.backwardElimAll ()                          // R^2, R^2 bar, R^2 cv
    val k = cols.size
	println("k:")
	println(k)
	
    println (s"k = $k, n = ${x.dim2}")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Regression", lines = true)

    println (s"rSq = $rSq")


} // RegressionTest17 object



//RegTest18 Tests stepwise regression with MLR for concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest18` object tests the `Regression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest18
 */
object RegressionTest18 extends App
{
    //Concrete code for MLR
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "Concrete_Data.csv", "Concrete", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val rg = new Regression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)

    val (cols, rSq) = rg.stepRegressionAll ()                          // R^2, R^2 bar, R^2 cv
    val k = cols.size
	println("k:")
	println(k)
	
    println (s"k = $k, n = ${x.dim2}")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Regression", lines = true)

    println (s"rSq = $rSq")


} // RegressionTest18 object


//LassoTest6 tests Lasso regression on MLR for concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LassoRegressionTest6` object tests the `LassoRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 Bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.LassoRegressionTest6
 */
object LassoRegressionTest6 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val rg = new LassoRegression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)
    val n = x.dim2                                                    // number of parameters/variables

    //banner ("Forward Selection Test")
    //val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 Bar, R^2 cv

    banner ("Forward Selection Test")
    val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 Bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")

} // LassoRegressionTest6 object


//RidgeReg8 tests Ridge regression on MLR for concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeRegressionTest8` object tests the `RidgeRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RidgeRegressionTest8
 */
object RidgeRegressionTest8 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val rrg = RidgeRegression (x, y, null, RidgeRegression.hp, Cholesky)
    println (rrg.analyze ().report)
    println (rrg.summary)
    val n = x.dim2                                                     // number of variables

    banner ("Forward Selection Test")
    val (cols, rSq) = rrg.forwardSelAll ()                             // R^2, R^2 bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                       // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for RidgeRegression", lines = true)

    println (s"rSq = $rSq")

} // RidgeRegressionTest8 object


//Quadtest10 tests forward selection on Quadratic regression for concrete
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest10` object tests the `QuadRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest10
 */
object QuadRegressionTest10 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete: QuadRegression")
    val qrg = new QuadRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                  // number of variables
    val nt = QuadRegression.numTerms (n)                             // number of terms
    println (qrg.summary)
    println (s"n = $n, nt = $nt")
    
    banner ("Forward Selection Test")
    val (cols, rSq) = qrg.forwardSelAll ()  
    val k = cols.size
    val t = VectorD.range (1, k)                                     // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for QuadRegression", lines = true)
    println (s"rSq = $rSq")

} // QuadRegressionTest10 object



//Quadtest11 tests backward elimination on Quadratic regression for concrete
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest11` object tests the `QuadRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest11
 */
object QuadRegressionTest11 extends App
{
    import scalation.columnar_db.Relation

   

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete: QuadRegression")
    val qrg = new QuadRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                  // number of variables
    val nt = QuadRegression.numTerms (n)                             // number of terms
    println (qrg.summary)
    println (s"n = $n, nt = $nt")


    banner ("Backward Elimination Test")
    val (cols, rSq) = qrg.backwardElimAll ()                           // R^2, R^2 bar, R^2 cv

    val k = cols.size
    val t = VectorD.range (1, k)                                     // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for QuadRegression", lines = true)
    println (s"rSq = $rSq")

} // QuadRegressionTest11 object


//Quadtest12 tests stepwise regression on Quadratic regression for concrete
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest12` object tests the `QuadRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest12
 */
object QuadRegressionTest12 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete: QuadRegression")
    val qrg = new QuadRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                  // number of variables
    val nt = QuadRegression.numTerms (n)                             // number of terms
    println (qrg.summary)
    println (s"n = $n, nt = $nt")

    banner ("Stepwise Regression Test")
    val (cols, rSq) = qrg.stepRegressionAll ()                           // R^2, R^2 bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                     // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for QuadRegression", lines = true)
    println (s"rSq = $rSq")

} // QuadRegressionTest12 object


//QuadXTest8 tests QuadX regression using forward selection on the concrete set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest8` object tests the `QuadXRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest8
 */
object QuadXRegressionTest8 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val qrg = new QuadXRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                // number of variables
    val nt = QuadXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (qrg.summary)

    banner ("Forward Selection Test")
    val (cols, rSq) = qrg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // QuadXRegressionTest8 object


//QuadXTest9 tests QuadX regression using backward elimination on the concrete set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest9` object tests the `QuadXRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest9
 */
object QuadXRegressionTest9 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val qrg = new QuadXRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                // number of variables
    val nt = QuadXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (qrg.summary)

    banner ("Backward Elimination Test")
    val (cols, rSq) = qrg.backwardElimAll()    
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // QuadXRegressionTest9 object


//QuadXTest10 tests QuadX regression using stepwise regression on the concrete set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest10` object tests the `QuadXRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest10
 */
object QuadXRegressionTest10 extends App
{
    import scalation.columnar_db.Relation

     banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val qrg = new QuadXRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                // number of variables
    val nt = QuadXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (qrg.summary)

    banner ("Stepwise Regression Test")
    val (cols, rSq) = qrg.stepRegressionAll()    
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // QuadXRegressionTest10 object























//CubicTest8 tests cubic regression using forward selection on concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest8` object tests the `CubicRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest8
 */
object CubicRegressionTest8 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)

//  import ExampleConcrete.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val crg = new CubicRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)

    banner ("Forward Selection Test")
    val (cols, rSq) = crg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv

    val k = cols.size - 1
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, lines = true)
    println (s"k = $k, nt = $nt")
    println (s"rSq = $rSq")

} // CubicRegressionTest8 object




//CubicTest9 tests cubic regression using backward elimination on concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest9` object tests the `CubicRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest9
 */
object CubicRegressionTest9 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)

//  import ExampleConcrete.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val crg = new CubicRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)

    banner ("Backward Elimination Test")
    val (cols, rSq) = crg.backwardElimAll ()                         // R^2, R^2 bar, R^2 cv

    val k = cols.size - 1
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, lines = true)
    println (s"k = $k, nt = $nt")
    println (s"rSq = $rSq")

} // CubicRegressionTest9 object

//CubicTest10 tests cubic regression using stepwise regression on concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest10` object tests the `CubicRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest10
 */
object CubicRegressionTest10 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)

//  import ExampleConcrete.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val crg = new CubicRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)


    banner ("Stepwise Regression Test")
    val (cols, rSq) = crg.stepRegressionAll () 
    val k = cols.size - 1
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, lines = true)
    println (s"k = $k, nt = $nt")
    println (s"rSq = $rSq")

} // CubicRegressionTest10 object



//cubicXtest8 tests forward selection on cubicX regression for concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest8` object tests the `CubicXRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest8
 */
object CubicXRegressionTest8 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val crg = new CubicXRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)

    banner ("Forward Selection Test")
    
    val (cols, rSq) = crg.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
   

    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // CubicXRegressionTest8 object



//cubicXtest9 tests backward elimination on cubicX regression for concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest9` object tests the `CubicXRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest9
 */
object CubicXRegressionTest9 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val crg = new CubicXRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)



    banner ("Backward Elimination Test")
    val (cols, rSq) = crg.backwardElimAll ()      

    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // CubicXRegressionTest9 object



//cubicXtest10 tests stepwise regression on cubicX regression for concrete
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest10` object tests the `CubicXRegression` class using the Concrete
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Concrete_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest10
 */
object CubicXRegressionTest10 extends App
{
    import scalation.columnar_db.Relation

    banner ("Concrete relation")
    val auto_tab = Relation (BASE_DIR + "Concrete_Data.csv", "Concrete", null, -1)
    //auto_tab.show ()

    banner ("Concrete (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Concrete regression")
    val crg = new CubicXRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)


    banner ("Stepwise Regression Test")
    val (cols, rSq) = crg.stepRegressionAll ()      

    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // CubicXRegressionTest10 object
