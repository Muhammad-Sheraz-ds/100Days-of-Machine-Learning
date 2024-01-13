# Module 07: Exploratory Data Analysis

## Univariate Analysis
- Histogram: `pyplot.hist()` in Matplotlib or `sns.histplot()` in Seaborn
- Bar chart: pyplot.bar() in Matplotlib or sns.countplot() in Seaborn
- Pie chart: pyplot.pie() in Matplotlib
- Line chart: pyplot.plot() in Matplotlib or sns.lineplot() in Seaborn
- Box plot: pyplot.boxplot() in Matplotlib or sns.boxplot() in Seaborn
- Violin plot: pyplot.violinplot() in Matplotlib or sns.violinplot() in Seaborn
- Kernel density estimation (KDE) plot: sns.kdeplot() in Seaborn
- Rug plot: sns.rugplot() in Seaborn
- Swarm plot: sns.swarmplot() in Seaborn
- Density plot: sns.kdeplot() in Seaborn
- Q-Q plot: sns.qqplot() in Seaborn
- Empirical cumulative distribution function (ECDF) plot: sns.ecdfplot() in Seaborn


## Bivariate Analysis

#### Numerical-Numerical Analysis

- Scatter plot: This plot is used to visualize the relationship between two numerical columns by plotting each data point as a point on a Cartesian coordinate system. It can be created using the `pyplot.scatter()` function in Matplotlib or the `sns.scatterplot()` function in Seaborn.

- Line plot: This plot is used to visualize the relationship between two numerical columns by connecting each data point with a line. It can be created using the `pyplot.plot()` function in Matplotlib or the `sns.lineplot()` function in Seaborn.

- Hexbin plot: This plot is similar to a scatter plot, but it groups data points into hexagonal bins and color codes the bins based on the number of points in each bin. It can be created using the `pyplot.hexbin()` function in Matplotlib or the `sns.jointplot()` function with `kind=hex` in Seaborn.

- Box plot: This plot is used to visualize the distribution of a numerical column by plotting the median, quartiles, and outliers of the data. It can be created using the `pyplot.boxplot()` function in Matplotlib or the `sns.boxplot()` function in Seaborn.

- Violin plot: This plot is similar to a box plot, but it shows the distribution of a numerical column as a kernel density estimate, which is a smoothed version of the histogram of the data. It can be created using the `pyplot.violinplot()` function in Matplotlib or the `sns.violinplot()` function in Seaborn.

- Heatmap: This plot is used to visualize the relationship between two numerical columns by plotting a grid of colored cells, where the color of each cell represents the value of the data point at that location. It can be created using the pyplot.imshow() function or `pyplot.matshow()` function in Matplotlib or the `sns.heatmap()` function in Seaborn.

- Jointplot: This plot is provided by seaborn library, it combines scatter plot with histograms, you can use it to visualize the relationship between two numerical columns, with the added benefit of seeing the distribution of each variable. It can be created using the `sns.jointplot()` function in Seaborn.


#### Numerical-Categorical Analysis
- Bar chart: This plot shows the mean or median of a numerical column for different categories of a categorical column. It can be created using the `pyplot.bar()` function in Matplotlib or the `sns.barplot()` function in Seaborn.

- Box plot: This plot shows the distribution of a numerical column for different categories of a categorical column by plotting the median, quartiles, and outliers of the data. It can be created using the `pyplot.boxplot()` function in Matplotlib or the `sns.boxplot()` function in Seaborn.

- Violin plot: This plot is similar to a box plot, but it shows the distribution of a numerical column for different categories of a categorical column as a kernel density estimate, which is a smoothed version of the histogram of the data. It can be created using the `pyplot.violinplot()` function in Matplotlib or the `sns.violinplot()` function in Seaborn.

- Point plot: This plot is similar to a bar chart, but it shows the mean or median of a numerical column for different categories of a categorical column with a confidence interval. It can be created using the `sns.pointplot()` function in Seaborn.

- Swarm plot: This plot is similar to a scatter plot, but it is used to show the distribution of a numerical column for different categories of a categorical column and prevent overplotting. It can be created using the `sns.swarmplot()` function in Seaborn.

- Factor plot: This plot is provided by seaborn library, it allows you to show the relationship between a numerical column and a categorical column, by allowing you to define the kind of plot you want and the columns to use, you can use it with multiple ways, boxplot, violinplot, etc.


#### Categorical-Categorical Analysis

- Bar chart: This plot shows the count or percentage of observations in each category of one categorical variable, grouped by the categories of the other categorical variable. It can be created using the `pyplot.bar()` function in Matplotlib or the `sns.countplot()` function in Seaborn.

- Stacked bar chart: This plot is similar to a bar chart, but it stacks the bars of the different categories of one categorical variable on top of each other, rather than showing them side by side. It can be created using the `pyplot.bar()` function in Matplotlib or the `sns.barplot()` function in Seaborn with hue parameter.

- Mosaic plot: This plot is a way of visualizing the relationship between two categorical variables by dividing the plot into rectangles, where the area of each rectangle represents the proportion of observations that fall into each combination of categories. It can be created using the `mosaic()` function in the `mosaicplot` library.

- Categorical scatter plot: This plot is a scatter plot where the points are colored or shaped according to the categories of one or more categorical variables. It can be created using the `sns.scatterplot()` function in seaborn.

- Categorical heatmap: This plot is a heatmap where the cells are colored according to the count or percentage of observations that fall into each combination of categories of two categorical variables. It can be created using the `sns.heatmap()` function in Seaborn with categorical parameter.

- Categorical boxplot: This plot is a boxplot where the boxplot is drawn for different categories of one categorical variable, grouped by the categories of the other categorical variable. It can be created using the `sns.boxplot()` function in Seaborn with hue parameter.


## Multi-Variate Analysis

- Scatter plot matrix (SPLOM): This plot shows scatter plots of all pairs of numerical variables in a dataset. It can be created using the `scatter_matrix()` function in the plotting module of the Pandas library or the `pairplot()` function in Seaborn.

- Parallel Coordinates: This plot allows to compare the multivariate distribution of a set of numerical variables, it shows the axis for each variable and the dots are the observations, it can be created using the `parallel_coordinates()` function in the plotting module of the Pandas library or the `parallel_coordinates()` function in the `mpl_toolkits.mplot3d` module of Matplotlib.

- Heatmap: This plot shows the relationship between two or more variables by plotting a grid of colored cells, where the color of each cell represents the value of the data point at that location. It can be created using the `pyplot.imshow()` function or `pyplot.matshow()` function in Matplotlib or the `sns.heatmap()` function in Seaborn.

- 3D Scatter plot: This plot shows the relationship between three numerical variables by plotting each data point as a point on a 3-dimensional Cartesian coordinate system. It can be created using the `pyplot.scatter()` function in Matplotlib with the `projection='3d'` parameter or the `scatter3D()` function in the `plotly.express` module.

- 3D Surface plot: This plot shows the relationship between three numerical variables by plotting a 3-dimensional surface of the data. It can be created using the `plot_surface()` function in the `mpl_toolkits.mplot3d` module of Matplotlib.

- Pair Grid plot: This plot is provided by seaborn library, it allows you to show different plots for different combinations of variables, you can use it to plot scatter plot, line plot, histograms, etc.