## Linear Regression with Tensorflow Example

### Creating the Conda Environment

```conda env create -f conda.yaml```

### Activating the Conda Environment

```conda activate fishlength```

### Running the Example

The run.py python file is the training script.

The run.py scripts needs two parameters

e.g.

```python run.py 1000 0.001```

- 1000 is the amount of steps to train
- 0.001 is the learning rate for the GradientDescentOptimizer


### Check the learning rate in the MLFlow

Start the MLFlow server with:

```mlflow server```

In the overview click select the experiment that you want to check. 

In the metrics section click on the loos link.

You should see something like this:

![Loss History](https://github.com/kstrempel/LinearRegressionFishLength/blob/master/loss.png)

Now you can play with the amount of steps and learning rate. :-)
