# Roles

Pranay - Helped with Scala and R, Report
Dylan - Headed Scala development, Report
Nathan - Headed R development, Report


# Running the Code


### Scala
We used scala and R for our project.

To run the scala files, take the included CSVs and paste them into the scalation_1.6/data/analytics/ folder so they can be accessed directly by the scala files.

The various scala files should just be copied into the scalation_1.6/scalation_modeling/src/main/scala/scalation/analytics/ folder. 

Then just run sbt from the scalation_modeling directory and run compile. Each scala file is associated with one dataset and all the objects inside of it pertain to the various Regression methods. To run one of them type runMain scalation.analytics.NameOfTest in the sbt commandline prompt.


### R

For the R files, we recommend using RStudio since it will make the installation of the packages much simpler. Open the files in RStudio and it should prompt you to download the required packages. 

Once the packages are installed, simply hit CTRL+SHIFT+ENTER to run the whole file or select a portion of the code and click run or hit CTRL+ENTER to run it. This should produce the related plots in the bottom right area of RStudio and you can click the arrows in that screen to switch between the plots. The text related output will be printed in the Console area at the bottom left. 