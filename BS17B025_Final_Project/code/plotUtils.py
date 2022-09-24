import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plt_target(y_true, y_pred, r2, mse, model_name='Regression', plt_style='seaborn-whitegrid', figsize=(12, 8)):
    '''
    A method that makes comparison plots between the 
    actual and predicted target values.

    Args: 
        y_true     -> (1D-numpy array) corresponding to the
                   actual value of the target variable.

        y_pred     -> (1D-numpy array) corresponding to the 
                   predictions made by the regression
                   model.

        r2         -> (float) R-squared statistical measure.

        mse        -> (float) Mean Squared Error.
        
        model_name  -> (str) represents the name of the
                    regression model that made the 
                    predictions.
    '''

    #Plotting using style of interest.
    with plt.style.context(plt_style):

        #Defining a figure.
        fig = plt.figure(figsize=figsize)

        plt.scatter(y_true, y_pred, color='darkcyan', alpha=0.6)

        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()

        plt.plot([xmin, xmax], [ymin, ymax], 'r--', linewidth=1, alpha=0.3)
        plt.xlabel('Actual value', fontsize=15)
        plt.ylabel('Predcted value', fontsize=15)

        plt.annotate(f'$R^2$ Score: {r2:.4}',
                        size=12,
                        xy=(xmin, ymax),
                        xytext=(10, -15),
                        textcoords='offset points')

        plt.annotate(f'MSE: {mse:.5}',
                        size=12,
                        wrap=True,
                        xy=(xmin, ymax),
                        xytext=(10, -35),
                        textcoords='offset points')

        plt.suptitle(f'Slowness in traffic predictions using {model_name}', fontsize=20)
        plt.show()

    return fig

def plt_residuals(y_true,y_pred, model_name='Regression', plt_style='seaborn-whitegrid', figsize=(18,6)):
    '''
    A method that makes the residual histogram and scatter
    plots of a fitted linear regression model.

    Args: 
        y_true     -> (1D-numpy array) corresponding to the
                   actual value of the target variable.

        y_pred     -> (1D-numpy array) corresponding to the 
                   predictions made by the regression
                   model.
        
        model_name  -> (str) represents the name of the
                    regression model that made the 
                    predictions.
    '''

    residuals = y_true - y_pred

    #Plotting using style of interest.
    with plt.style.context(plt_style):

        #Defining a figure.
        fig,axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        fig.suptitle(f'Residual plots of {model_name}', fontsize=20)
        axes = axes.ravel()

        #Histogram plot of the residual.
        axes[0].set_title('Histogram')

        axes[0] = sns.histplot(residuals,ax=axes[0], kde='True', color='darkcyan')
        axes[0].axvline(x=np.mean(residuals), color='red', alpha=0.3, label='mean')
        axes[0].axvline(x=np.mean(residuals), color='orange', alpha=0.3, label='median')

        axes[0].set_xlabel('Residuals', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend()

        #Scatter plot of the residual.
        axes[1].set_title('Scatter plot')

        scatter_kws = {'alpha':0.6}
        line_kws = {'color':'red', 'lw':1, 'alpha':0.3}

        axes[1] = sns.residplot(x=y_pred,
                                y=y_true, 
                                ax=axes[1], 
                                lowess=True,
                                color='darkcyan',
                                scatter_kws=scatter_kws,
                                line_kws=line_kws)

        axes[1].set_xlabel('Predictions', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)

    return fig
