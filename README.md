1. Summary

Traditional press brakes rely on fixed formulas, which makes them sensitive to variations such as thickness changes, coated materials, and tool wear.
This project improves pressure prediction by redesigning preprocessing, refining feature selection, and performing structured hyperparameter optimization.

The dataset contains 360 real bending samples across multiple materials, thicknesses, bending angles, and valid tool–die combinations.

SVR was selected as the primary model due to its stability, while GBDT was used as a secondary model for comparison and future scalability.

2. Methodology
2.1 Data Preprocessing

- Fixed the previous data-leakage issue by splitting before preprocessing.

- All interpolation, scaling, and encoding are fit on training data only.

- Test data uses the same fitted parameters.

2.2 Feature Selection

- Removed tool/die IDs (they are deterministic to thickness and do not directly affect pressure).

- Retained only physically meaningful features to simplify the model and reduce noise.

2.3 Hyperparameter Optimization

- Replaced manual trial-and-error with grid search + k-fold cross-validation.

- Tuned key parameters:

--  SVR Parameter:kernel, C, epsilon, gamma, shrinking, cache_size, max_iter, tol

--  GBDT Parameter: n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, random_state

3. Results
SVR

- Lower test error

- Smaller train–test gap

- More stable error distribution

- Better generalization after fixing preprocessing

GBDT

- Smoother learning curves

- Reduced variance and tighter error distribution

- Improved performance after structured tuning

Physical Validation

- Real press-brake tests still show deviations due to factors not captured in the dataset (material batches, springback, tooling wear, alignment).

Overall:
SVR remains the most stable model; GBDT shows potential for future datasets with richer or more complex features.

4. Future Work

- Collect larger and more diverse datasets

- Include material hardness, surface conditions, and tool-wear indicators

- Test ensemble and neural-network models

- Integrate sensor-based monitoring for real-time prediction

