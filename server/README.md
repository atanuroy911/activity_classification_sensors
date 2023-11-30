# Activity_Pred_ML

Random Forest Model: https://1drv.ms/v/s!Ao6HNgs77kddgbYg9453UssNtrFUtA?e=EQ7Swr


Static Dataset: https://1drv.ms/u/s!Ao6HNgs77kddgbYixQNO6uDNxw-Z5w?e=9rq4cP

```python
python3 prepare_dataset.py --i datasets/out/aruba/data --o ARUBA_Preprocessed --w 25
```

```python
python3 generate_subsets.py --i ARUBA_Preprocessed_25_padded --p subsets
```

```python
python3 main.py --i ARUBA_Preprocessed_25_padded --d subsets/ --c batch_1024_epoch_400_early_stop_mask_0.0 --m LSTM
```