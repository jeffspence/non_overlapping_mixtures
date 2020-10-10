This repo contains all of the scripts used in
"Flexible mean field variational inference
using mixtures of non-overlapping exponential families."

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
            against the truth. The results are saved in data/ldpred/
            The usage is:

                ```python ldpred.py <sigma_sq_e> <number_of_repetitions>```

            The outputs are ```data/ldpred/cor_mat_<sigma_sq_e>.txt``` that contain
            the correlations and ```mse_mat_<sigma_sq_e>.txt``` that contain
            the MSE for each VI scheme.


## ldpred_plotter.R

            Takes the output of ldpred.py and produces figures
            like Figure 1 and saves them in ```figs/```.


## sparse_pca.py
            Analogous to ```ldpred.py```, this simulates data for the
            sparse PCA model as described in the main text and
            then runs the various VI schemes on the results.
            The results (and the raw data) are saved in data/sparse_pca/
            The usage is:
                python sparse_pca.py <num_non_zero_variables> <rep_name>
            In the paper num_non_zero_variables was always 100
            and rep_name was used to generate 6 datasets.  rep_name
            is optional.
            The ouputs are:
                data[_rep<rep_name>].npy:
                    The simulated data matrix
                keep[_rep<rep_name>].npy:
                    The indices of the non-zero effect sizes
                vi_mu_w_<sigma_sq_0>[_rep<rep_name>].npy:
                    mu_w as defined in the appendix for the naive VI
                    schemes
                vi_mu_w_sparse[_rep<rep_name>].npy:
                    mu_w as defined in the appendix for the scheme
                    based on the non-overlapping mixtures trick.
                vi_mu_z_<sigma_sq_0>[_rep<rep_name>].npy:
                    Same as above but for mu_z.
                vi_mu_z_sparse[_rep<rep_name>].npy:
                    Same as above but for mu_z.
                vi_psi_0_sparse[_rep<rep_name>].npy:
                    psi_0 as defined in the appendix for the scheme
                    based on the non-overlapping mixtures trick.
                vi_sigma_w_sparse[_rep<rep_name>].npy:
                    sigma_w as defined in the appendix for the scheme
                    based on the non-overlapping mixtures trick.

## PCA_plotter_notebook.ipynb

            Jupyter notebook to visualize and plot the figures in
            Figure 2 (additional formatting done in inkscape).


## make_pca_plots.py

            Takes the data produced by sparse_pca.py and makes figures
            like those in Figure 5.  Assumes that the <rep_name> above
            are 1, 2, ..., 5.  Outputs files with self-explanatory names
            in figs/


## threshold_plotter.R

            Produces the plot in Figure 3 (additional text was added
            in inkscape).


## data/

        All of the outputs from scripts in the code directory
        and all of the necessary data to make the plots in
        the figs directory. The data are described above as ouputs
        of ldpred.py and sparse_pca.py
        Due to the size constraints on NeurIPS submissions,
        the sparse_pca folder will be empty until sparse_pca.py
        is run.  Due to randomness in the simulation process
        the results will not exactly match those in the paper
        but will be comparable.


## figs
        All of the raw figures used in the manuscript.
        Note that labels were often added directly to figures
        or figures were combined into multi-panel figures using
        inkscape (inkscape.org).
