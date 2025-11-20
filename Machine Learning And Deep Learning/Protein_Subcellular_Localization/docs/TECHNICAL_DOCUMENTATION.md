# Technical Documentation

## System Architecture and Mathematical Formulations

### 1. Image Preprocessing

#### 1.1 Normalization
For a TIFF image $I$ with pixel values in range $[I_{min}, I_{max}]$, normalization to $[0, 1]$ is performed as:

$$I_{norm}(x, y) = \frac{I(x, y) - I_{min}}{I_{max} - I_{min}}$$

For 16-bit images:
$$I_{norm} = \frac{I}{65535}$$

For 8-bit images:
$$I_{norm} = \frac{I}{255}$$

#### 1.2 Resize Operation
Images are resized to target dimensions $(H_{target}, W_{target})$ using bilinear interpolation:

$$I_{resized}(x', y') = \sum_{i,j} I(i, j) \cdot w(x'-i) \cdot w(y'-j)$$

where $w$ is the interpolation kernel.

### 2. Cellpose Segmentation

Cellpose predicts:
1. **Flows** $F = (F_x, F_y)$: Vector field pointing toward cell centers
2. **Cell probability** $P$: Probability of pixel being inside a cell

The segmentation mask $M$ is obtained by:
1. Following flows to find cell centers
2. Thresholding cell probability: $P(x,y) > \tau$
3. Assigning each pixel to nearest cell center

Parameters:
- Diameter: $d$ (expected cell size)
- Flow threshold: $\tau_f = 0.4$
- Cell probability threshold: $\tau_p = 0.0$

### 3. Superpixel Generation (SLIC)

SLIC (Simple Linear Iterative Clustering) generates superpixels by clustering pixels in the 5D space $[l, a, b, x, y]$ where:
- $(l, a, b)$: Color in LAB space
- $(x, y)$: Spatial coordinates

Distance metric:
$$D = \sqrt{d_{lab}^2 + \left(\frac{d_{xy}}{S}\right)^2 m^2}$$

where:
- $d_{lab} = \sqrt{(l_i - l_j)^2 + (a_i - a_j)^2 + (b_i - b_j)^2}$
- $d_{xy} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$
- $S = \sqrt{N/K}$: Grid interval
- $m$: Compactness parameter

### 4. Feature Extraction

For each superpixel $s$, we extract:

#### 4.1 Intensity Features
- Mean: $\mu_s = \frac{1}{|s|} \sum_{(x,y) \in s} I(x,y)$
- Std: $\sigma_s = \sqrt{\frac{1}{|s|} \sum_{(x,y) \in s} (I(x,y) - \mu_s)^2}$
- Max: $I_{max}^s = \max_{(x,y) \in s} I(x,y)$
- Min: $I_{min}^s = \min_{(x,y) \in s} I(x,y)$

#### 4.2 Geometric Features
- Area: $A_s = |s|$
- Perimeter: $P_s$
- Eccentricity: $e_s = \sqrt{1 - \frac{\lambda_{min}}{\lambda_{max}}}$
- Centroid: $C_s = (\bar{x}_s, \bar{y}_s)$

Feature vector:
$$\mathbf{f}_s = [\mu_s, \sigma_s, I_{max}^s, I_{min}^s, A_s, P_s, e_s, \bar{x}_s, \bar{y}_s]$$

### 5. Graph Construction

#### 5.1 Adjacency Graph
Graph $G = (V, E)$ where:
- $V = \{s_1, s_2, ..., s_n\}$: Superpixels
- $E = \{(s_i, s_j) | s_i \text{ adjacent to } s_j\}$

Adjacency defined by 4-connectivity in the segmentation map.

#### 5.2 k-NN Graph
Connect each node to its $k$ nearest neighbors based on feature similarity:
$$d(s_i, s_j) = ||\mathbf{f}_{s_i} - \mathbf{f}_{s_j}||_2$$

### 6. CNN Model (VGG16)

#### 6.1 Architecture
VGG16 consists of 13 convolutional layers and 3 fully connected layers:

$$\text{Conv Block} = \text{Conv2D} \rightarrow \text{ReLU} \rightarrow \text{MaxPool}$$

Final classification layer:
$$y_{CNN} = \text{softmax}(W^T h + b)$$

where $h$ is the feature vector from the last hidden layer.

#### 6.2 Loss Function
Cross-entropy loss:
$$\mathcal{L}_{CNN} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

where:
- $N$: Number of samples
- $C$: Number of classes
- $y_{i,c}$: True label (one-hot)
- $\hat{y}_{i,c}$: Predicted probability

### 7. Graph Neural Network

#### 7.1 Graph Convolutional Network (GCN)
Layer-wise propagation:
$$H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})$$

where:
- $\tilde{A} = A + I$: Adjacency with self-loops
- $\tilde{D}$: Degree matrix of $\tilde{A}$
- $H^{(l)}$: Node features at layer $l$
- $W^{(l)}$: Learnable weight matrix
- $\sigma$: Activation function (ReLU)

#### 7.2 GraphSAGE
Aggregation and update:
$$h_v^{(l+1)} = \sigma(W^{(l)} \cdot \text{CONCAT}(h_v^{(l)}, \text{AGG}(\{h_u^{(l)}, \forall u \in \mathcal{N}(v)\})))$$

Aggregation function:
$$\text{AGG} = \text{MEAN}, \text{MAX}, \text{ or } \text{LSTM}$$

#### 7.3 Graph Attention Network (GAT)
Attention mechanism:
$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [W h_i || W h_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [W h_i || W h_k]))}$$

Node update:
$$h_i^{(l+1)} = \sigma(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W^{(l)} h_j^{(l)})$$

#### 7.4 Global Pooling
Graph-level representation:
$$h_G = \frac{1}{|V|} \sum_{v \in V} h_v^{(L)}$$

Classification:
$$y_{GNN} = \text{softmax}(W_c h_G + b_c)$$

### 8. Model Fusion

#### 8.1 Weighted Average
$$P_{fused}(c) = \alpha \cdot P_{CNN}(c) + (1-\alpha) \cdot P_{GNN}(c)$$

where:
- $P_{CNN}(c)$: CNN probability for class $c$
- $P_{GNN}(c)$: GNN probability for class $c$
- $\alpha \in [0, 1]$: Fusion weight (default: 0.6)

Prediction:
$$\hat{y} = \arg\max_c P_{fused}(c)$$

#### 8.2 Voting
Hard voting based on individual predictions:
$$\hat{y} = \text{mode}(\hat{y}_{CNN}, \hat{y}_{GNN})$$

### 9. Evaluation Metrics

#### 9.1 Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

#### 9.2 Precision
$$\text{Precision} = \frac{TP}{TP + FP}$$

#### 9.3 Recall (Sensitivity)
$$\text{Recall} = \frac{TP}{TP + FN}$$

#### 9.4 F1-Score
$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### 9.5 Specificity
$$\text{Specificity} = \frac{TN}{TN + FP}$$

#### 9.6 Macro-Average
$$\text{Metric}_{macro} = \frac{1}{C} \sum_{c=1}^{C} \text{Metric}_c$$

where $C$ is the number of classes.

### 10. System Pipeline

```
Input TIFF Image
      ↓
[Preprocessing]
  - Normalize
  - Resize
      ↓
[Segmentation] ← Cellpose
  - Detect regions
  - Extract masks
      ↓
[Dual Processing Path]
      ├─────────────────┬─────────────────┐
      ↓                 ↓                 ↓
[CNN Branch]     [Superpixel Gen]   [Feature Extract]
  VGG16               SLIC           Intensity+Geometry
      ↓                 ↓                 ↓
[CNN Predict]    [Graph Construct]  [Graph Features]
      ↓                 ↓                 ↓
      ↓            [GNN Model]            ↓
      ↓          GCN/SAGE/GAT            ↓
      ↓                 ↓                 ↓
      ↓           [GNN Predict]          ↓
      ↓                 ↓                 ↓
      └─────────────────┴─────────────────┘
                        ↓
                  [Model Fusion]
                 Weighted Average
                        ↓
                 [Final Prediction]
                        ↓
              [Visualization & Report]
```

## Hyperparameters

### CNN (VGG16)
- Input size: 224 × 224 × 3
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 50
- Optimizer: Adam
- Frozen layers: 15

### GNN
- Hidden dimensions: 64
- Number of layers: 3
- Dropout: 0.5
- Learning rate: 0.001
- Epochs: 100
- Optimizer: Adam

### Superpixels (SLIC)
- Number of segments: 100
- Compactness: 10

### Fusion
- CNN weight: 0.6
- GNN weight: 0.4

## Computational Complexity

### CNN
- Forward pass: $O(n \cdot m \cdot k^2 \cdot c_{in} \cdot c_{out})$
- Training: $O(B \cdot E \cdot n \cdot m \cdot k^2 \cdot c)$

where:
- $n \times m$: Image dimensions
- $k$: Kernel size
- $c_{in}, c_{out}$: Input/output channels
- $B$: Batch size
- $E$: Epochs

### GNN
- Forward pass: $O(|E| \cdot d \cdot h + |V| \cdot h^2)$
- Training: $O(E_{epochs} \cdot (|E| \cdot d \cdot h + |V| \cdot h^2))$

where:
- $|V|$: Number of nodes
- $|E|$: Number of edges
- $d$: Input feature dimension
- $h$: Hidden dimension

## References

1. Stringer, C. et al. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature Methods.
2. Simonyan, K. & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.
3. Kipf, T. N. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
4. Hamilton, W. et al. (2017). Inductive Representation Learning on Large Graphs.
5. Veličković, P. et al. (2018). Graph Attention Networks.
6. Achanta, R. et al. (2012). SLIC Superpixels Compared to State-of-the-Art Superpixel Methods.
