from typing import Tuple, List, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prPlots(betas: List[float], userLambda: float, x_train0: pd.DataFrame, x_train: pd.DataFrame, data: pd.DataFrame, model: Any, bivariate_inputs: List[int], n_steps: int = 15, sd_scale: int = 2, method: str = "dirac", device: str = "cpu") -> None:
  """
  Generate partial response plots based on the selected lambda and model coefficients.

  Parameters
  ----------
  betas : List[float]
      Model coefficients.
  userLambda : float
      Selected lambda value.
  x_train0 : pd.DataFrame
      Original training dataset before any transformations.
  x_train : pd.DataFrame
      Transformed training dataset.
  data : pd.DataFrame
      Combined dataset used for training/testing.
  model : Any
      The trained model.
  bivariate_inputs : List[int]
      Indices of features to be used for bivariate analysis.
  n_steps : int, optional
      Number of steps to use for generating plots.
  sd_scale : int, optional
      Scaling factor for standard deviation in the data normalization.
  method : str, optional
      Method used to compute the partial responses ('dirac' or 'lebesgue').
  device : str, optional
      Device to run the model computation ('cpu' or 'cuda').

  Returns
  -------
  None
  """
  if method.lower() == "dirac":

    x0 = np.zeros((1,x_train.shape[1]))
    if isinstance(model,list) == True:
      y0 = mlpmask_pred(x0,model,device=device)
    elif hasattr(model,'predict_proba'):
      y0 = model.predict_proba(x0)[:,1]
    else:
      y0 = model.predict(x0,device=device)

    logit_y0 = np.log(y0/(1-y0))

    # plot all selected partial responses
    for pr in np.where(abs(betas[:,userLambda])>0.1)[0]:

      # univariate partial response plots ----------------------------------------------------------------------------------------------------------------- univariate plots
      if pr < x_train.shape[1]:
        print(pr,"- univ")

        x_step0 = np.linspace(min(x_train0.iloc[:,pr]),max(x_train0.iloc[:,pr]),n_steps)
        x_step = np.linspace(min(x_train.iloc[:,pr]),max(x_train.iloc[:,pr]),n_steps)

        x_in = np.zeros([n_steps,x_train.shape[1]])
        x_in[:,pr] = x_step
        if isinstance(model,list) == True:
          pred_y_xi = mlpmask_pred(x_in,model,device=device)
        elif hasattr(model,'predict_proba'):
          pred_y_xi = model.predict_proba(x_in)[:,1]
        else:
          pred_y_xi = model.predict(x_in, device=device)

        y_xi = np.log(pred_y_xi/(1-pred_y_xi))-logit_y0

        fig, ax1 = plt.subplots()
        ax1.set_title("Partial Response Plot")
        ax1.set_xlabel(x_train0.columns[pr])
        ax1.set_ylabel("Contribution to logit",color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Frequency")

        # pr plot for discrete data
        if len(x_train0.iloc[:,pr].unique()) < n_steps:
          ax2.bar(np.sort(x_train0.iloc[:,pr].unique()),x_train0.iloc[:,pr].value_counts().sort_index(),facecolor="none",ec="black")
        # pr plot for continuous data
        else:
          ax2.hist(x_train0.iloc[:,pr],histtype="bar",facecolor="none",ec="black",bins=n_steps)
        ax1.plot((x_step*(x_train0.iloc[:,pr].std()*2))+x_train0.iloc[:,pr].median(),y_xi,color="red")
        plt.show()

      # bivariate partial response plots ----------------------------------------------------------------------------------------------------------------------- bivariate plots
      else:
        pr_i = int(bivariate_inputs[pr-x_train.shape[1],0])
        pr_j = int(bivariate_inputs[pr-x_train.shape[1],1])

        if (len(x_train.iloc[:,pr_i].unique()) < n_steps) != (len(x_train.iloc[:,pr_j].unique()) < n_steps):
          if len(x_train.iloc[:,pr_i].unique()) < n_steps:
            x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
            x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)
          else:
            x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
            x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

        elif len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
          x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
          x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

        else:
          x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
          x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)

        y_xij = np.zeros([len(x_step0_j),len(x_step0_i)])

        for i in range(0,len(x_step0_i)):
          for j in range(0,len(x_step0_j)):
            x_in = np.zeros(x_train.shape[1])
            x_in[pr_i] = (x_step0_i[i]-x_train0.iloc[:,pr_i].median())/(x_train0.iloc[:,pr_i].std()*2)
            x_in[pr_j] = (x_step0_j[j]-x_train0.iloc[:,pr_j].median())/(x_train0.iloc[:,pr_j].std()*2)
            if isinstance(model,list) == True:
              pred_y_xij = mlpmask_pred(x_in.reshape(1,len(x_in)),model,device=device)
            elif hasattr(model,'predict_proba'):
              pred_y_xij = model.predict_proba(x_in.reshape(1,len(x_in)))[:,1]
            else:
              # pred_y_xij = model.predict(x_in, device=device.reshape(1,len(x_in)))
              pred_y_xij = model.predict(x_in.reshape(1,len(x_in)), device=device)

            y_xij[j,i] = np.log(pred_y_xij[0][0]/(1-pred_y_xij[0][0]))-logit_y0

# ------------------------------------------------------------------------------------------------------------------------------------------------ mixed responses
        if (len(x_train.iloc[:,pr_i].unique()) < n_steps) != (len(x_train.iloc[:,pr_j].unique()) < n_steps):

          fig, ax1 = plt.subplots()
          ax1.set_title("Partial Response Plot")

          ax1.set_ylabel("Contribution to logit",color="red")
          ax2 = ax1.twinx()
          ax2.set_ylabel("Frequency")
          colourmap = plt.get_cmap('seismic_r')
          if len(x_train.iloc[:,pr_i].unique()) < n_steps:
            for i in range(0,len(x_train.iloc[:,pr_i].unique())):
              ax1.set_xlabel(x_train0.columns[pr_j])
              ax1.plot(x_step0_j,y_xij[:,i],label=np.sort(x_train0.iloc[:,pr_i].unique())[i])
              ax1.legend(title=x_train0.columns[pr_i])
          else:
            for j in range(0,len(x_train.iloc[:,pr_j].unique())):
              ax1.set_xlabel(x_train0.columns[pr_i])
              ax1.plot(x_step0_i,y_xij[j,:],label=np.sort(x_train0.iloc[:,pr_j].unique())[j])
              ax1.legend(title=x_train0.columns[pr_j])

          plt.show()

        else:
          fig = plt.figure()
          ax = plt.axes()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------- categorical/categorical
          if len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
            heatmap = ax.imshow(y_xij,cmap="viridis", aspect="auto")
            print(x_step0_i)
            ax.set_xticks(list(range(0,len(x_step0_i))),labels=np.round(x_step0_i,2))
            ax.set_yticks(list(range(0,len(x_step0_j))),labels=np.round(x_step0_j,2))
            boxSettings = dict(boxstyle='round', facecolor='grey', alpha=0.5)
            for i in range(0,len(x_step0_i)):
              for j in range(0,len(x_step0_j)):
                ax.text(i, j, np.round(y_xij[j, i],2),ha="center", va="center", color="w",bbox=boxSettings)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------- continuous/continuous
          else:
            X, Y = np.meshgrid(x_step0_i.reshape(len(x_step0_i),1), x_step0_j.reshape(len(x_step0_j),1))
            contour_heatmap = ax.contourf(X,Y,y_xij)
            c2 = plt.contour(X,Y,y_xij, cmap='Greys')
            ax.clabel(c2, inline=True, fontsize=10)
            ax.set_xticks(x_step0_i,labels=np.round(x_step0_i,2),rotation=45)
            ax.set_yticks(x_step0_j,labels=np.round(x_step0_j,2))
            fig.colorbar(contour_heatmap, orientation='vertical')
          ax.set_xlabel(x_train.columns[pr_i])
          ax.set_ylabel(x_train.columns[pr_j])

          plt.show()

#---------------------------------------------------------------------------------------------------------LEBESGUE MEASURE-------------------------------------------------------------------------------------------------------

  if method.lower() == "lebesgue":

    if isinstance(model,list) == True:
      y0 = mlpmask_pred(x_train,model,device=device)
    elif hasattr(model,'predict_proba'):
      y0 = model.predict_proba(x_train)[:,1]
    else:
      y0 = model.predict(x_train, device=device)

    logit_y0 = np.mean(np.log(y0/(1-y0)))

    # plot all selected partial responses
    for pr in np.where(abs(betas[:,userLambda])>0.1)[0]:

      # univariate parital response plots ----------------------------------------------------------------------------------------------------------------- univariate plots
      if pr < x_train.shape[1]:

        x_step0 = np.linspace(min(x_train0.iloc[:,pr]),max(x_train0.iloc[:,pr]),n_steps)
        x_step = np.linspace(min(x_train.iloc[:,pr]),max(x_train.iloc[:,pr]),n_steps)
        y_xi = np.zeros(len(x_step))

        for k in range(0,len(x_step)):

          x_in = x_train.copy()
          x_in.iloc[:,pr] = x_step[k]

          if isinstance(model,list) == True:
            pred_y_xi = mlpmask_pred(x_in,model,device=device)
          elif hasattr(model,'predict_proba'):
            pred_y_xi = model.predict_proba(x_in)[:,1]
          else:
            pred_y_xi = model.predict(x_in, device=device)

          y_xi[k] = np.mean(np.log(pred_y_xi/(1-pred_y_xi)))-logit_y0

        fig, ax1 = plt.subplots()
        ax1.set_title("Partial Response Plot")
        ax1.set_xlabel(x_train0.columns[pr])
        ax1.set_ylabel("Contribution to logit",color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Frequency")

        # pr plot for discrete data
        if len(x_train0.iloc[:,pr].unique()) < n_steps:
          ax2.bar(np.sort(x_train0.iloc[:,pr].unique()),x_train0.iloc[:,pr].value_counts().sort_index(),facecolor="none",ec="black")
        # pr plot for continuous data
        else:
          ax2.hist(x_train0.iloc[:,pr],histtype="bar",facecolor="none",ec="black",bins=n_steps)
        ax1.plot((x_step*(x_train0.iloc[:,pr].std()*2))+x_train0.iloc[:,pr].median(),y_xi,color="red")
        plt.show()

      # bivariate partial response plots ----------------------------------------------------------------------------------------------------------------------- bivariate plots
      else:
        pr_i = int(bivariate_inputs[pr-x_train.shape[1],0])
        pr_j = int(bivariate_inputs[pr-x_train.shape[1],1])

        if (len(x_train.iloc[:,pr_i].unique()) < n_steps) != (len(x_train.iloc[:,pr_j].unique()) < n_steps):
          if len(x_train.iloc[:,pr_i].unique()) < n_steps:
            x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
            x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)
          else:
            x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
            x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

        elif len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
          x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
          x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

        else:
          x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
          x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)


        y_xij = np.zeros([len(x_step0_j),len(x_step0_i)])

        for i in range(0,len(x_step0_i)):
          for j in range(0,len(x_step0_j)):
            x_in = x_train.copy()
            x_in.iloc[:,pr_i] = (x_step0_i[i]-x_train0.iloc[:,pr_i].median())/(x_train0.iloc[:,pr_i].std()*2)
            x_in.iloc[:,pr_j] = (x_step0_j[j]-x_train0.iloc[:,pr_j].median())/(x_train0.iloc[:,pr_j].std()*2)

            if isinstance(model,list) == True:
              pred_y_xij = mlpmask_pred(x_in,model,device=device)
            elif hasattr(model,'predict_proba'):
              pred_y_xij = model.predict_proba(x_in)[:,1]
            else:
              pred_y_xij = model.predict(x_in, device=device)

            y_xij[j,i] = np.log(np.mean(pred_y_xij)/(1-np.mean(pred_y_xij)))-logit_y0


# ------------------------------------------------------------------------------------------------------------------------------------------------ mixed responses
        if (len(x_train.iloc[:,pr_i].unique()) < n_steps) != (len(x_train.iloc[:,pr_j].unique()) < n_steps):

          fig, ax1 = plt.subplots()
          ax1.set_title("Partial Response Plot")

          ax1.set_ylabel("Contribution to logit",color="red")
          ax2 = ax1.twinx()
          ax2.set_ylabel("Frequency")
          colourmap = plt.get_cmap('seismic_r')
          if len(x_train.iloc[:,pr_i].unique()) < n_steps:
            for i in range(0,len(x_train.iloc[:,pr_i].unique())):
              ax1.set_xlabel(x_train0.columns[pr_j])
              ax1.plot(x_step0_j,y_xij[:,i],label=np.sort(x_train0.iloc[:,pr_i].unique())[i])
              ax1.legend(title=x_train0.columns[pr_i])
          else:
            for j in range(0,len(x_train.iloc[:,pr_j].unique())):
              ax1.set_xlabel(x_train0.columns[pr_i])
              ax1.plot(x_step0_i,y_xij[j,:],label=np.sort(x_train0.iloc[:,pr_j].unique())[j])
              ax1.legend(title=x_train0.columns[pr_j])

          plt.show()

        else:
          fig = plt.figure()
          ax = plt.axes()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------- categorical/categorical
          if len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
            heatmap = ax.imshow(y_xij,cmap="viridis", aspect="auto")
            ax.set_xticks(list(range(0,len(x_step0_i))),labels=np.round(x_step0_i,2))
            ax.set_yticks(list(range(0,len(x_step0_j))),labels=np.round(x_step0_j,2))
            boxSettings = dict(boxstyle='round', facecolor='grey', alpha=0.5)
            for i in range(0,len(x_step0_i)):
              for j in range(0,len(x_step0_j)):
                ax.text(i, j, np.round(y_xij[j, i],2),ha="center", va="center", color="w",bbox=boxSettings)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------- continuous/continuous
          else:

            X, Y = np.meshgrid(x_step0_i.reshape(len(x_step0_i),1), x_step0_j.reshape(len(x_step0_j),1))
            contour_heatmap = ax.contourf(X,Y,y_xij)
            c2 = plt.contour(X,Y,y_xij, cmap='Greys')
            ax.clabel(c2, inline=True, fontsize=10)
            ax.set_xticks(x_step0_i,labels=np.round(x_step0_i,2),rotation=45)
            ax.set_yticks(x_step0_j,labels=np.round(x_step0_j,2))
            fig.colorbar(contour_heatmap, orientation='vertical')
          ax.set_xlabel(x_train.columns[pr_i])
          ax.set_ylabel(x_train.columns[pr_j])
          plt.show()
