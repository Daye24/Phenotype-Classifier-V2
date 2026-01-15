# Data Directory

Place your dataset CSV files here.

## Expected Format

Your `dataset.csv` should have:
- Feature columns (numeric or categorical)
- A `label` column for the target variable

Example:
```csv
feature1,feature2,feature3,label
1.2,3.4,5.6,class_a
2.3,4.5,6.7,class_b
...
```

## Files

- `dataset.csv` - Main training dataset
- `new_data.csv` - Data for predictions (optional)

**Note:** Large CSV files are ignored by git (see `.gitignore`)
