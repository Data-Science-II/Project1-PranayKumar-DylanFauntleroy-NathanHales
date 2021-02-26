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

//RegTest19 Tests forward selection with MLR for red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest19` object tests the `Regression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest19
 */
object RegressionTest19 extends App
{
    //Red Wine code for MLR
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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


} // RegressionTest19 object




//RegTest20 Tests backward elimination with MLR for red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest20` object tests the `Regression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest20
 */
object RegressionTest20 extends App
{
    //Red Wine code for MLR
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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


} // RegressionTest20 object



//RegTest21 Tests stepwise regression with MLR for red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest21` object tests the `Regression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest21
 */
object RegressionTest21 extends App
{
    //Red Wine code for MLR
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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


} // RegressionTest21 object


//LassoTest7 tests Lasso regression on MLR for red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LassoRegressionTest7` object tests the `LassoRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 Bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.LassoRegressionTest7
 */
object LassoRegressionTest7 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // LassoRegressionTest7 object


//RidgeReg9 tests Ridge regression on MLR for red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeRegressionTest9` object tests the `RidgeRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RidgeRegressionTest9
 */
object RidgeRegressionTest9 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // RidgeRegressionTest9 object


//Quadtest13 tests forward selection on Quadratic regression for red wine
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest13` object tests the `QuadRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest13
 */
object QuadRegressionTest13 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine: QuadRegression")
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

} // QuadRegressionTest13 object



//Quadtest14 tests backward elimination on Quadratic regression for red wine
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest14` object tests the `QuadRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest14
 */
object QuadRegressionTest14 extends App
{
    import scalation.columnar_db.Relation

   

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine: QuadRegression")
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

} // QuadRegressionTest14 object


//Quadtest15 tests stepwise regression on Quadratic regression for red wine
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest15` object tests the `QuadRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest15
 */
object QuadRegressionTest15 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine: QuadRegression")
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

} // QuadRegressionTest15 object


//QuadXTes11 tests QuadX regression using forward selection on the red wine set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest11` object tests the `QuadXRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest11
 */
object QuadXRegressionTest11 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // QuadXRegressionTest11 object


//QuadXTest12 tests QuadX regression using backward elimination on the red wine set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest12` object tests the `QuadXRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest12
 */
object QuadXRegressionTest12 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // QuadXRegressionTest12 object


//QuadXTest13 tests QuadX regression using stepwise regression on the red wine set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest13` object tests the `QuadXRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest13
 */
object QuadXRegressionTest13 extends App
{
    import scalation.columnar_db.Relation

     banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // QuadXRegressionTest13 object




//CubicTest11 tests cubic regression using forward selection on red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest11` object tests the `CubicRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest11
 */
object CubicRegressionTest11 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)

//  import ExampleRed Wine.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // CubicRegressionTest11 object




//CubicTest12 tests cubic regression using backward elimination on red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest12` object tests the `CubicRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest12
 */
object CubicRegressionTest12 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)

//  import ExampleRed Wine.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // CubicRegressionTest12 object

//CubicTest13 tests cubic regression using stepwise regression on red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest13` object tests the `CubicRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest13
 */
object CubicRegressionTest13 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)

//  import ExampleRed Wine.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // CubicRegressionTest13 object



//cubicXtest11 tests forward selection on cubicX regression for red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest11` object tests the `CubicXRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest11
 */
object CubicXRegressionTest11 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // CubicXRegressionTest11 object



//cubicXtest12 tests backward elimination on cubicX regression for red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest12` object tests the `CubicXRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest12
 */
object CubicXRegressionTest12 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // CubicXRegressionTest12 object



//cubicXtest13 tests stepwise regression on cubicX regression for red wine
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest13` object tests the `CubicXRegression` class using the Red Wine
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Red_Wine_Quality.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest13
 */
object CubicXRegressionTest13 extends App
{
    import scalation.columnar_db.Relation

    banner ("Red Wine relation")
    val auto_tab = Relation (BASE_DIR + "Red_Wine_Quality.csv", "Red Wine", null, -1)
    //auto_tab.show ()

    banner ("Red Wine (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 10, 11)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Red Wine regression")
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

} // CubicXRegressionTest13 object

