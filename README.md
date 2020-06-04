# Kohonen Network with Wine Dataset
[Dataset](http://archive.ics.uci.edu/ml/datasets/wine)

## Kohonen Network
Kohnen Network使用[@alexarnimueller](https://github.com/alexarnimueller)所撰寫的[Source Code](https://github.com/alexarnimueller/som)
> Source Code Ref: Kohonen, T. Self-Organized Formation of Topologically Correct Feature Maps. Biol. Cybern. 1982, 43 (1), 59–69.

## Feature Normalizatoion
進行資料特徵歸一化，為了更佳的分群效果
```python
def feature_normalizatoion(X):
    for i in range(len(X)):
        x = X[i]
        x_sqare_sum_root = np.sum(x**2)**0.5
        x = [f/x_sqare_sum_root for f in x]
        X[i] = x
    return X
```

## Kohonen Network Training
- network size : 8*8
- epoch : 10000
- learning rate : `lr = 1 / (1 + (epoch / 0.5) **4)`

<img src="https://github.com/p208p2002/kohonen-network-with-wine-dataset/blob/master/images/som_error.png?raw=true" width="400px"/>

## Results

### 網路權重熱力圖
<img src="https://github.com/p208p2002/kohonen-network-with-wine-dataset/blob/master/images/distance_map.png?raw=true" width="400px"/>

### 各別類別權重熱力圖
<div style='display:inline'>
<img src="https://github.com/p208p2002/kohonen-network-with-wine-dataset/blob/master/images/class_1.png?raw=true" width="200px"/>

<img src="https://github.com/p208p2002/kohonen-network-with-wine-dataset/blob/master/images/class_2.png?raw=true" width="200px"/>

<img src="https://github.com/p208p2002/kohonen-network-with-wine-dataset/blob/master/images/class_3.png?raw=true" width="200px"/>
</div>

## 分群結果視覺化
<img src="https://github.com/p208p2002/kohonen-network-with-wine-dataset/blob/master/images/som.png?raw=true" width="400px"/>
</div>
