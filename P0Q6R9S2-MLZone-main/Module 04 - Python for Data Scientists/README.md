# Module 4: Python for Data Scientist

## NumPy:
- Introduction to Numpy Library. Array Creation using np.array() constructor. Miscellaneous Array Creation Functions: np.zeros(), np.ones(), np.empty(), np.full(), np.eye(), np.fromstring(), np.arange(), np.linspace(), np.random.rand(), np.random.randint(), np.zeros_like().
- A Comparison between Python Lists, Python Arrays and NumPy Arrays. Memory Consumption of Python List and Numpy Array. Operation cost on Python List and Numpy Array.
- NumPy universal functions (unary and binary), Scalar Math, Arithmetic Operations, Aggregate Functions, Comparison Operations. Searching NumPy Arrays.
- Slicing NumPy Arrays. Boolean and Fancy Array Indexing.
- Broadcasting NumPy Arrays. Reshaping NumPy Arrays using the shape attribute, np.reshape(), np.resize(), ndarray.transpose(), np.swapaxes(), np.flatten() methods. Sorting Arrays using np.sort(). Iterating NumPy Arrays.
- Updating existing values of NumPy array elements. Append new elements to a NumPy array using np.append(). Insert new elements in a NumPy array using np.insert(). Delete elements of a NumPy array using np.delete().
- Alias vs Shallow Copy vs Deep Copy.
- Array concatenation, stacking and splitting.
- Reading and Writing Numeric data from text/csv Files to NumPy Arrays.

## Pandas:
- Overview of Pandas Library and its Data Structures. Install Pandas Library. Read Datasets into Pandas Dataframe. Python Dictionaries vs Pandas Dataframes. Anatomy of a Dataframe
- Creating a Series
	- From Python List
	- From NumPy Arrays
	- From Python Dictionary
	- From a scalar value
- Attributes of a Pandas Series. Understanding Index in a Series and its usage: (Identification, Selection, Subsetting, Alignment).
- Anatomy of a Dataframe, Creating Dataframe from an empty dataframe, from Two-Dimensional NumPy Array, from Dictionary of Python Lists, from Dictionary of Panda Series. Attributes of a Dataframe
- Reading data from a CSV/TSV File Reading a CSV file from a Remote System. Writing Contents of Dataframe to a CSV File. Reading data from an EXCEL File. Writing Contents of Dataframe to an EXCEL File. Reading data from a JSON File. Writing Contents of Dataframe to a JSON File.
- Understanding Indices of a Dataframe, Selecting Row(s) and Column(s) of a Dataframe using `df[]`, Selecting Rows and Columns using `iloc` Method, Selecting Rows and Columns using `loc` Method, Conditional Selection, Selecting columns of a specific data type
- Modifying Column labels of Dataframe. Modifying Row indices of Dataframe. Modifying Row(s) Data (Records) of a Dataframe. Modifying multiple Rows using `map()`, `df.remove()`, `df.apply()`, `df.applymap()` methods.
- Reading a Sample Dataframe, Adding a New Column or Deleting an Existing Column from a Dataframe, Adding a New Row or Deleting an Existing Row(s) from a Dataframe, Adding a New Column with Conditional Values, Deleting Row(s) Based on Specific Condition, Delete a Column Based on Specific Condition, Change Datatype of a Pandas Series, Sorting dataframes using `df.sort_values()`, `df.sort_index()` methods.
- Have an insight about the Dataset, Identify Column(s) containing Null/Missing values using df.isna() method, Handle/Impute the Null/Missing Values under the Column using `df.loc`, Handle Missing values under a Numeric/Categorical Column using fillna() method, Handle Repeating Values (for same information), Create a new Column by Modifying an Existing Column, Delete Rows containing NaN values using df.dropna() method, Convert Categorical Variables into Numerical.
- Overview of Aggregation Functions and the `agg()` method, Computing the Minimum Temperature of different Cities using hard way, groupby methods. Practice GroupBy on Stack Overflow Survey Dataset.
- Merging DataFrames using pd.merge() method, Perform Inner Join (which is default), Perform Outer/Full Outer Join, Perform Left Outer Join, Perform Right Outer Join, Additional Parameters to pd.merge() Method, Concatenation using pd.concat() method, Appending Dataframes using df.append() method.
- Reshape Data Using `pivot()` , `pivot_table()` , `melt()` and `crosstab()` methods.
- Recap of Python's Built-in Time and Datetime Modules, Overview of Pandas Time Series Data Structures, Converting Strings to Pandas DateTime64 type, Convert a Scalar String to DateTime, Convert Pandas Series to DateTime, Handling Issues of DateTime Formats, Convert a Single Integer to Pandas DateTime, Practicing with a Simple Dataset, UFO Dataset, and Crypto-Currency Dataset, Concept of Up Sampling and Down Sampling, Bonus Task.

## EDA and Visualization using Seaborn & Matplotlib:
- Overview of Data Visualization, Choosing an appropriate Chart for the problem at hand, Anatomy of a Figure, Download and Install Matplotlib, Programming with Matplotlib, How to draw a Line Chart, Enhance the Graph Step by Step and Saving it on Disk. Understanding Sub-Plots.
- Understanding Bar-plot, Visualizing Bar-Plot for Stack Overflow Dataset, Understanding Scatter-plot, Visualizing Scatter Plot for Houses Dataset.
- Understanding Pie Chart, Visualizing Pie Chart for Stack Overflow Survey Dataset, Understanding Histogram, Visualizing Histogram for Stack Overflow Survey Dataset, Visualizing Histogram of an Image.
- Recap of previous session, Understanding/Visualizing Box Plot, Violin Plot, Heatmap
- Creating Multiple Sub-Plots within a Figure Object.
- Overview of Seaborn Library, Downloading and Importing Seaborn Library, Built-in Datasets of Seaborn Library, Car_Crashes, Flights, Tips, Iris, Titanic Datasets. Three Figure Level Methods of Seaborn. Using sns.relplot() method to draw line and scatter plots. Using sns.catplot() method to draw bar, count, box, violin, strip, and swarm plots. Using sns.displot() method to draw hist, kde and ecdf plots.