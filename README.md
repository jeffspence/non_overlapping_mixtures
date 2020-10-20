This repo contains all of the scripts used in
["Flexible mean field variational inference
using mixtures of non-overlapping exponential families."](https://arxiv.org/abs/2010.06768)

It also contains plotting scripts, although with different
seeds, the output of the scripts may be slightly different
from that in the paper.  Overall, however, the results will
be largely comparable.  To make the finalized figures in the
paper, I sometimes directly added labels to the figures, or
combined the figures in [inkscape](https://inkscape.org)
but the data in the figures were in no way manipulated.

Below I describe each of the scripts in the ```code/``` directory.
Some of these scripts will generate data in the ```data/``` directory
and some of the scripts will use that data to generate figures.
All of the scripts are assumed to be run from the ```code``` directory.


## ldpred.py

Simulates data as described in the main text and then
computes the correlation and MSE of the point estimates
against the truth. The results are saved in ```data/ldpred/```
The usage is:

```python ldpred.py <sigma_sq_e> <number_of_repetitions>```

The outputs are ```data/ldpred/cor_mat_<sigma_sq_e>.txt``` that contain
the correlations and ```mse_mat_<sigma_sq_e>.txt``` that contain
the MSE for each VI scheme.  Additionally, it saves the raw data
to be used with ```nimble_code.R``` and ```pyro_code_discrete.py```.
The plotting scripts assume that ```<number_of_repetitions>``` is 100.


## nimble_code.R

Performs MCMC on the ldpred model data simulated by ```ldpred.py```.
The usage is:

```Rscript nimble_code.R <1-indexed rep> <sigma_sq_e>```

The output is
 ```data/ldpred/nimble_results_<1-indexed rep>_<sigma_sq_e>.txt```,
 which contrains the correlation and MSE for this run.  These runs can
then be combined using ```parse_nimble.py```.  The plotting scripts
assume that the ```<1-indexed rep>```s run from 1-100.


## parse_nimble.py

Combines all of the outputs from ```nimble_code.R```, so that
they can be used in ```ldpred_plotter.R```.


## pyro_code_discrete.py

Performs boosting black box VI using ```pyro``` on the ldpred
model data simulated by ```ldpred.py```. The usage is:

```python pyro_code_discrete.py <1-indexed rep> <sigma_sq_e>```

The output is
```data/ldpred/pyro_discrete_<sigma_sq_e>_rep_<1-indexed rep>_cor.txt```
and
```data/ldpred/pyro_discrete_<sigma_sq_e>_rep_<1-indexed rep>_mse.txt```
containing the correlation of the posterior mean and true betas and MSE
respectively.  The plotting scripts assume that the ``<1-indexed rep>```s
run from 1-100.

## parse_pyro.py

Combines all of the outputs from ```pyro_code_discrete.py```, so that
they can be used in ```ldpred_plotter.R```.


## ldpred_plotter.R

Takes the output of ```ldpred.py```, ```parse_nimble.py```,
and ```parse_pyro.py```  and produces figures
like Figure 1 and saves them in ```figs/```.


## sparse_pca.py
Analogous to ```ldpred.py```, this simulates data for the
sparse PCA model as described in the main text and
then runs the various VI schemes on the results.
The results (and the raw data) are saved in data/sparse_pca/
The usage is:

```python sparse_pca.py <num_non_zero_variables> <rep_name>```

In the paper num_non_zero_variables was always 100
and rep_name was used to generate 6 datasets.  rep_name
is optional.
The ouputs are:

```data/sparse_pca/data[_rep<rep_name>].npy```: The simulated data matrix

```data/sparse_pca/keep[_rep<rep_name>].npy```: The indices of the non-zero effect sizes

```data/sparse_pca/vi_mu_w_<sigma_sq_0>[_rep<rep_name>].npy```:
mu_w as defined in the appendix for the naive VI
schemes

```data/sparse_pca/vi_mu_w_sparse[_rep<rep_name>].npy```:
mu_w as defined in the appendix for the scheme
based on the non-overlapping mixtures trick.
                
```data/sparse_pca/vi_mu_z_<sigma_sq_0>[_rep<rep_name>].npy```:
Same as above but for mu_z.
                
```data/sparse_pca/vi_mu_z_sparse[_rep<rep_name>].npy```:
Same as above but for mu_z.
                
```data/sparse_pca/vi_psi_0_sparse[_rep<rep_name>].npy```:
psi_0 as defined in the appendix for the scheme
based on the non-overlapping mixtures trick.
                
```data/sparse_pca/vi_sigma_w_sparse[_rep<rep_name>].npy```:
sigma_w as defined in the appendix for the scheme
based on the non-overlapping mixtures trick.


## PCA_plotter_notebook.ipynb

Jupyter notebook to visualize and plot the figures in
Figure 2 (additional formatting done in inkscape).


## reconstruction_error.py

Reads in reps 1, 2, ..., 5 and prints the min, mean,
and max reconstruction error.


## make_pca_plots.py

Takes the data produced by sparse_pca.py and makes figures
like those in Figure 5.  Assumes that the <rep_name> above
are 1, 2, ..., 5.  Outputs files with self-explanatory names
in ```figs/```


## threshold_plotter.R

Produces the plot in Figure 3 (additional text was added
in inkscape).


## data/

All of the outputs from scripts in the code directory
and all of the necessary data to make the plots in
the figs directory. The data are described above as ouputs
of ldpred.py and sparse_pca.py
Due to the size constraints on GitHub,
this folder will be empty until the above scripts are 
run.  Due to randomness in the simulation process
the results will not exactly match those in the paper
but will be comparable.


## figs/

All of the raw figures used in the manuscript.
Note that labels were often added directly to figures
or figures were combined into multi-panel figures using
[inkscape](https://inkscape.org).


## Real Data Application
Data downloading and preprocessing were performed essentially exactly
as described in in the [kallisto](https://www.kallistobus.tools) tutorial implemented
in [this Google Collab notebook](https://colab.research.google.com/github/pachterlab/kallistobustools/blob/master/notebooks/kb_analysis_0_python.ipynb).
To facilitate reproduction, large portions of this notebook
are copied into ```code/real_data_PBMC_dataset.ipynb```.  Running the notebook should
reproduce the figures in the paper.
