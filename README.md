# Recommendation System using Hierarchical Poisson Factorization 

The purpose of this project is to recommend relevant items to users based on their previous ratings. For this task, I chose Fashion Products data set, which can be found at the following address: https://www.kaggle.com/datasets/bhanupratapbiswas/fashion-products.

Hierarchical Poisson Factorization algorithm aims, via a combination of Gamma and Poisson distributions, to predict a set of ratings for each user in a matrix of `N` users x `N` items, and based on those ratings one can model the recommendations. 

The algorithm inspired from [this paper](https://arxiv.org/abs/1311.1704) looks as follows:

```
Initialize the user parameters for each user u:
- sample activity: ξ_u ~ Gamma(a', a'/b')
- sample preference for each component k: θ_uk ~ Gamma(a, ξ_u)

Initialize the item parameters for each item i:
- sample popularity: η_i ~ Gamma(c', c'/d')
- sample attribute for each component k: β_ik ~ Gamma(c, η_i)

Sample rating for each user u and item i: y_ui ~ Gamma(Transpose(θ_uk) • β_ik)

```

A visual representation of the hyperparameter tuning and more details about the implementation are found in files `sanity_check_and_grid_search.ipynb` and `recommend_all_data.ipynb`.
