t-Distributed Stochastic Neighbor Embedding (t-SNE) is a powerful, non-linear technique primarily used for exploratory data analysis and visualizing high-dimensional data. It was developed by Laurens van der Maaten and Geoffrey Hinton in 2008. Here's an overview of how t-SNE works and why it's particularly useful for data visualization tasks:

### Purpose of t-SNE

t-SNE is designed to reduce high-dimensional datasets to lower-dimensional spaces (typically 2D or 3D) that are suitable for human observation and intuitive interpretation. The technique is particularly effective at revealing structures at multiple scales, which makes it an excellent tool for exploring the intrinsic clustering behaviors of complex datasets.

### How t-SNE Works

1. **Similarity Computation in High-Dimensional Space**: 
   - t-SNE starts by converting the high-dimensional Euclidean distances between points into conditional probabilities that represent similarities. The similarity of datapoint $x_j$ to datapoint $x_i$ is the conditional probability $p_{j|i}$ that $x_i$ would pick $x_j$ as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at $x_i$.
   - This probability is high for close points and low for points that are far apart. For the point $x_i$, this process is mathematically represented as:
    $$
     p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}
    $$
   - The bandwidth of the Gaussian kernel $\sigma_i$ is adapted for each datapoint to capture the local density around that point (perplexity parameter).

2. **Symmetrization**: 
   - To simplify the computation and to make it more robust, $p_{ij}$ is defined as the symmetrized version of the conditional probabilities:
    $$
     p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}
    $$
   - Here, $N$ is the total number of data points.

3. **Similarity Computation in Low-Dimensional Space**:
   - In the low-dimensional map, t-SNE defines a similar probability but using a simpler Student-t distribution (which has heavier tails) to measure similarities between points $y_i$ and $y_j$:
    $$
     q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}
    $$

4. **Cost Function (Kullback-Leibler Divergence)**:
   - t-SNE aims to minimize the divergence between these two distributions $P$ (high-dimensional) and $Q$ (low-dimensional) using a cost function known as the Kullback-Leibler divergence:
    $$
     C = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}
    $$
   - Gradient descent methods are typically employed to minimize this divergence.

### Key Features and Considerations

- **Perplexity**: A key hyperparameter for t-SNE, which loosely measures how to balance attention between local and global aspects of your data. Typical values are between 5 and 50.
- **Crowding Problem**: t-SNE's cost function is not convex, meaning different runs or slight changes in hyperparameters (like perplexity) can result in different outputs. This can also lead to the “crowding problem” where the dimensionality reduction space is too small to accommodate all similar points properly.
- **Random Initialization**: The positions of the points in the map are initialized randomly, meaning different runs of the algorithm can yield different results unless the random seed is fixed.
- **Computational Cost**: t-SNE can be computationally expensive and does not scale well to very large datasets.

t-SNE is best used as a technique for exploratory data analysis in a machine learning workflow, allowing users to visually assess patterns like clusters which might not be evident in high-dimensional spaces. It is not typically used for feature reduction prior to feeding data into a predictive model, partly because the t-SNE embedding is not necessarily consistent between different subsets of data.


$$
\begin{array}{l}
  \text{Para muitas dimensões: } \\\\
  p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}

  \\\\\\

  \text{Em baixas dimensões: } \\\\
  q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}

  \\\\\\
  
  \text{Função de custo: } \\\\
  C = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}
\end{array}
$$

$$
$$


$$
$$
