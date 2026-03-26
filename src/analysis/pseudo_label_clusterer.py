"""
Pseudo-label clustering with XGBoost refinement.

Pipeline:
1. Run an unsupervised base clustering method to obtain pseudo-labels.
2. Train an XGBoost classifier on those pseudo-labels.
3. Refine labels using one of several strategies:
   - direct classification
   - iterative self-training
   - confidence-based re-clustering
   - anomaly-aware refinement with synthetic negatives

The preprocessor accepts mixed feature tables and converts them into a numeric
matrix by:
- preserving numeric columns
- converting booleans to integers
- converting datetimes to integers
- one-hot encoding categorical columns
- imputing numeric missings with medians
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from .algo_clusterer import AlgoClusterer, ClusterMethod, ScalerType

logger = logging.getLogger(__name__)


class PseudoLabelStrategy(Enum):
    """Supported pseudo-label refinement strategies."""

    NONE = "none"
    DIRECT = "direct"
    SELF_TRAINING = "self_training"
    CONFIDENCE = "confidence_refinement"
    ANOMALY = "anomaly"


@dataclass
class PseudoLabelClusteringResult:
    """Output of pseudo-label clustering plus supervised refinement."""

    base_method: ClusterMethod
    strategy: PseudoLabelStrategy
    base_labels: pd.Series
    refined_labels: pd.Series
    class_probabilities: pd.DataFrame
    confidence: pd.Series
    feature_importance: pd.Series
    anomaly_score: pd.Series
    model: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class PseudoLabelClusterer:
    """Base clustering plus XGBoost pseudo-label refinement."""

    def __init__(
        self,
        base_method: ClusterMethod = ClusterMethod.GMM,
        n_clusters: int = 8,
        strategy: PseudoLabelStrategy = PseudoLabelStrategy.DIRECT,
        random_state: int = 42,
        scaler_type: ScalerType = ScalerType.ROBUST,
        confidence_threshold: float = 0.8,
        max_iter: int = 5,
        convergence_tol: float = 0.01,
        synthetic_ratio: float = 1.0,
        anomaly_quantile: float = 0.95,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        eps: float = 0.5,
        min_samples: int = 5,
        min_cluster_size: int = 15,
        n_jobs: int = -1,
    ):
        self.base_method = base_method
        self.n_clusters = n_clusters
        self.strategy = strategy
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.confidence_threshold = confidence_threshold
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.synthetic_ratio = synthetic_ratio
        self.anomaly_quantile = anomaly_quantile
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.n_jobs = n_jobs

    def fit_predict(self, features: pd.DataFrame) -> PseudoLabelClusteringResult:
        """Run base clustering plus the selected refinement strategy."""
        encoded, prep_meta = self._prepare_features(features)
        base_result = self._run_base_clustering(encoded)
        base_labels = pd.Series(base_result.labels, index=encoded.index, name="base_label", dtype=int)

        fallback_method = None
        if (base_labels >= 0).sum() == 0 and self.base_method in {ClusterMethod.DBSCAN, ClusterMethod.HDBSCAN}:
            logger.warning(
                "Base clustering with %s produced only noise; falling back to GMM for pseudo-label seeding",
                self.base_method.value,
            )
            fallback_method = ClusterMethod.GMM
            base_result = self._run_base_clustering(encoded, method_override=fallback_method)
            base_labels = pd.Series(base_result.labels, index=encoded.index, name="base_label", dtype=int)

        base_probs = self._base_probability_frame(base_labels)
        base_conf = base_probs.max(axis=1) if not base_probs.empty else pd.Series(0.0, index=encoded.index)
        empty_importance = pd.Series(dtype=float)
        empty_anomaly = pd.Series(np.nan, index=encoded.index, name="anomaly_score", dtype=float)

        metadata = {
            "base_method": self.base_method.value,
            "strategy": self.strategy.value,
            "base_n_clusters": int(base_result.n_clusters),
            "base_n_noise": int(base_result.n_noise),
            "base_silhouette": float(base_result.silhouette),
            "prepared_numeric_features": prep_meta["n_numeric_features"],
            "prepared_categorical_features": prep_meta["n_categorical_features"],
            "prepared_total_features": prep_meta["n_output_features"],
        }
        if fallback_method is not None:
            metadata["fallback_base_method"] = fallback_method.value

        if self.strategy == PseudoLabelStrategy.NONE:
            return PseudoLabelClusteringResult(
                base_method=self.base_method,
                strategy=self.strategy,
                base_labels=base_labels,
                refined_labels=base_labels.copy(),
                class_probabilities=base_probs,
                confidence=base_conf.rename("confidence"),
                feature_importance=empty_importance,
                anomaly_score=empty_anomaly,
                model=base_result.model,
                metadata=metadata,
            )

        if (base_labels >= 0).sum() == 0:
            logger.warning("Base clustering produced no non-noise labels; returning base labels")
            metadata["refinement_skipped"] = "no_non_noise_labels"
            return PseudoLabelClusteringResult(
                base_method=self.base_method,
                strategy=self.strategy,
                base_labels=base_labels,
                refined_labels=base_labels.copy(),
                class_probabilities=base_probs,
                confidence=base_conf.rename("confidence"),
                feature_importance=empty_importance,
                anomaly_score=empty_anomaly,
                model=base_result.model,
                metadata=metadata,
            )

        if self.strategy == PseudoLabelStrategy.DIRECT:
            refined = self._refine_direct(encoded, base_labels)
        elif self.strategy == PseudoLabelStrategy.SELF_TRAINING:
            refined = self._refine_self_training(encoded, base_labels)
        elif self.strategy == PseudoLabelStrategy.CONFIDENCE:
            refined = self._refine_confidence(encoded, base_labels)
        elif self.strategy == PseudoLabelStrategy.ANOMALY:
            refined = self._refine_anomaly(encoded, base_labels)
        else:
            raise ValueError(f"Unsupported pseudo-label strategy: {self.strategy}")

        refined.metadata = metadata | refined.metadata
        return refined

    def _prepare_features(self, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """Convert mixed feature tables into a numeric matrix."""
        if features is None or features.empty:
            raise ValueError("Cannot pseudo-label cluster an empty feature table")

        data = features.copy()
        numeric_frames = []
        categorical_frames = []
        numeric_cols = []
        categorical_cols = []

        for col in data.columns:
            series = data[col]
            if is_datetime64_any_dtype(series):
                converted = series.view("int64").replace({pd.NaT.value: np.nan}).astype(float)
                numeric_frames.append(converted.rename(col))
                numeric_cols.append(col)
            elif is_bool_dtype(series):
                numeric_frames.append(series.astype(int).rename(col))
                numeric_cols.append(col)
            elif is_numeric_dtype(series):
                numeric_frames.append(pd.to_numeric(series, errors="coerce").rename(col))
                numeric_cols.append(col)
            elif is_categorical_dtype(series) or series.dtype == object:
                categorical_frames.append(series.astype("string").fillna("__missing__").rename(col))
                categorical_cols.append(col)
            else:
                coerced = pd.to_numeric(series, errors="coerce")
                if coerced.notna().any():
                    numeric_frames.append(coerced.rename(col))
                    numeric_cols.append(col)
                else:
                    categorical_frames.append(series.astype("string").fillna("__missing__").rename(col))
                    categorical_cols.append(col)

        if numeric_frames:
            numeric_df = pd.concat(numeric_frames, axis=1)
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
            numeric_df = numeric_df.fillna(numeric_df.median())
            numeric_df = numeric_df.fillna(0.0)
        else:
            numeric_df = pd.DataFrame(index=data.index)

        if categorical_frames:
            cat_df = pd.concat(categorical_frames, axis=1)
            cat_encoded = pd.get_dummies(cat_df, prefix=cat_df.columns, prefix_sep="=", dtype=float)
        else:
            cat_encoded = pd.DataFrame(index=data.index)

        encoded = pd.concat([numeric_df, cat_encoded], axis=1)
        encoded = encoded.astype(float)

        return encoded, {
            "n_numeric_features": len(numeric_cols),
            "n_categorical_features": len(categorical_cols),
            "n_output_features": encoded.shape[1],
        }

    def _run_base_clustering(self, encoded: pd.DataFrame, method_override: ClusterMethod | None = None):
        """Run the unsupervised base clustering."""
        clusterer = AlgoClusterer(
            method=method_override or self.base_method,
            n_clusters=min(self.n_clusters, max(2, len(encoded) - 1)) if len(encoded) > 2 else 1,
            features=list(encoded.columns),
            random_state=self.random_state,
            scaler_type=self.scaler_type,
            eps=self.eps,
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
        )
        return clusterer.fit(encoded)

    def _refine_direct(
        self,
        encoded: pd.DataFrame,
        base_labels: pd.Series,
    ) -> PseudoLabelClusteringResult:
        """Train one XGBoost classifier on base pseudo-labels."""
        labels, probs, confidence, model, importance = self._fit_direct_classifier(encoded, base_labels)
        return PseudoLabelClusteringResult(
            base_method=self.base_method,
            strategy=self.strategy,
            base_labels=base_labels,
            refined_labels=labels.rename("family"),
            class_probabilities=probs,
            confidence=confidence.rename("confidence"),
            feature_importance=importance,
            anomaly_score=pd.Series(np.nan, index=encoded.index, name="anomaly_score", dtype=float),
            model=model,
            metadata={
                "refinement_iterations": 1,
                "n_final_clusters": int(labels[labels >= 0].nunique()),
            },
        )

    def _refine_self_training(
        self,
        encoded: pd.DataFrame,
        base_labels: pd.Series,
    ) -> PseudoLabelClusteringResult:
        """Iteratively retrain XGBoost on its own predictions until convergence."""
        history = []
        working_labels = base_labels.copy()
        prev_labels = None
        final_model = None
        final_probs = self._base_probability_frame(base_labels)
        final_conf = final_probs.max(axis=1) if not final_probs.empty else pd.Series(0.0, index=encoded.index)
        final_importance = pd.Series(dtype=float)

        for iteration in range(1, self.max_iter + 1):
            previous_labels = working_labels.copy()
            labels, probs, confidence, model, importance = self._fit_direct_classifier(encoded, working_labels)
            if prev_labels is None:
                change_ratio = 1.0
            else:
                change_ratio = float((labels != previous_labels).mean())

            history.append({
                "iteration": iteration,
                "change_ratio": change_ratio,
                "mean_confidence": float(confidence.mean()),
            })

            prev_labels = previous_labels
            working_labels = labels.astype(int)
            final_model = model
            final_probs = probs
            final_conf = confidence
            final_importance = importance

            if iteration > 1 and change_ratio <= self.convergence_tol:
                break

        return PseudoLabelClusteringResult(
            base_method=self.base_method,
            strategy=self.strategy,
            base_labels=base_labels,
            refined_labels=working_labels.rename("family"),
            class_probabilities=final_probs,
            confidence=final_conf.rename("confidence"),
            feature_importance=final_importance,
            anomaly_score=pd.Series(np.nan, index=encoded.index, name="anomaly_score", dtype=float),
            model=final_model,
            metadata={
                "refinement_iterations": len(history),
                "history": history,
                "n_final_clusters": int(working_labels[working_labels >= 0].nunique()),
            },
        )

    def _refine_confidence(
        self,
        encoded: pd.DataFrame,
        base_labels: pd.Series,
    ) -> PseudoLabelClusteringResult:
        """Re-cluster low-confidence points and retrain until stable."""
        labels, probs, confidence, model, importance = self._fit_direct_classifier(encoded, base_labels)
        working_labels = labels.astype(int)
        history = []

        for iteration in range(1, self.max_iter + 1):
            confident_mask = confidence >= self.confidence_threshold
            uncertain_mask = ~confident_mask

            history.append({
                "iteration": iteration,
                "n_confident": int(confident_mask.sum()),
                "n_uncertain": int(uncertain_mask.sum()),
                "mean_confidence": float(confidence.mean()),
            })

            if uncertain_mask.sum() < 2 or confident_mask.sum() < 2:
                break

            classes = sorted(int(c) for c in working_labels.loc[confident_mask].unique() if c >= 0)
            if len(classes) < 2:
                break

            uncertain_encoded = encoded.loc[uncertain_mask]
            n_local_clusters = min(len(classes), max(2, min(len(uncertain_encoded) - 1, self.n_clusters)))
            if n_local_clusters < 2:
                break

            uncertain_clusterer = AlgoClusterer(
                method=self.base_method,
                n_clusters=n_local_clusters,
                features=list(uncertain_encoded.columns),
                random_state=self.random_state,
                scaler_type=self.scaler_type,
                eps=self.eps,
                min_samples=self.min_samples,
                min_cluster_size=min(self.min_cluster_size, max(2, len(uncertain_encoded))),
            )
            uncertain_result = uncertain_clusterer.fit(uncertain_encoded)
            uncertain_labels = pd.Series(
                uncertain_result.labels,
                index=uncertain_encoded.index,
                name="uncertain_label",
                dtype=int,
            )

            aligned_labels = self._align_cluster_labels(
                confident_X=encoded.loc[confident_mask],
                confident_labels=working_labels.loc[confident_mask],
                uncertain_X=uncertain_encoded,
                uncertain_labels=uncertain_labels,
                fallback_labels=working_labels.loc[uncertain_mask],
            )

            new_working_labels = working_labels.copy()
            new_working_labels.loc[uncertain_mask] = aligned_labels
            label_change_ratio = float((new_working_labels != working_labels).mean())
            working_labels = new_working_labels

            labels, probs, confidence, model, importance = self._fit_direct_classifier(encoded, working_labels)
            working_labels = labels.astype(int)

            history[-1]["label_change_ratio"] = label_change_ratio
            if label_change_ratio <= self.convergence_tol:
                break

        return PseudoLabelClusteringResult(
            base_method=self.base_method,
            strategy=self.strategy,
            base_labels=base_labels,
            refined_labels=working_labels.rename("family"),
            class_probabilities=probs,
            confidence=confidence.rename("confidence"),
            feature_importance=importance,
            anomaly_score=pd.Series(np.nan, index=encoded.index, name="anomaly_score", dtype=float),
            model=model,
            metadata={
                "refinement_iterations": len(history),
                "history": history,
                "confidence_threshold": float(self.confidence_threshold),
                "n_final_clusters": int(working_labels[working_labels >= 0].nunique()),
            },
        )

    def _refine_anomaly(
        self,
        encoded: pd.DataFrame,
        base_labels: pd.Series,
    ) -> PseudoLabelClusteringResult:
        """Flag outliers with a synthetic-negative XGBoost detector."""
        labels, probs, confidence, model, importance = self._fit_direct_classifier(encoded, base_labels)
        anomaly_model, anomaly_score = self._fit_synthetic_anomaly_detector(encoded)
        anomaly_threshold = float(anomaly_score.quantile(self.anomaly_quantile))

        refined_labels = labels.copy()
        refined_labels.loc[anomaly_score >= anomaly_threshold] = -1

        return PseudoLabelClusteringResult(
            base_method=self.base_method,
            strategy=self.strategy,
            base_labels=base_labels,
            refined_labels=refined_labels.rename("family"),
            class_probabilities=probs,
            confidence=confidence.rename("confidence"),
            feature_importance=importance,
            anomaly_score=anomaly_score.rename("anomaly_score"),
            model={
                "classifier": model,
                "anomaly_detector": anomaly_model,
            },
            metadata={
                "refinement_iterations": 1,
                "anomaly_quantile": float(self.anomaly_quantile),
                "anomaly_threshold": anomaly_threshold,
                "n_anomalies": int((anomaly_score >= anomaly_threshold).sum()),
                "n_final_clusters": int(refined_labels[refined_labels >= 0].nunique()),
            },
        )

    def _fit_direct_classifier(
        self,
        encoded: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame, pd.Series, Any, pd.Series]:
        """Fit one XGBoost classifier on non-noise labels and predict all rows."""
        labelled_mask = labels >= 0
        classes = sorted(int(c) for c in labels.loc[labelled_mask].unique())
        if not classes:
            empty_probs = self._base_probability_frame(labels)
            empty_conf = empty_probs.max(axis=1) if not empty_probs.empty else pd.Series(0.0, index=encoded.index)
            return labels.copy(), empty_probs, empty_conf, None, pd.Series(dtype=float)

        if len(classes) == 1:
            only_class = classes[0]
            probs = pd.DataFrame(1.0, index=encoded.index, columns=[only_class])
            confidence = pd.Series(1.0, index=encoded.index, name="confidence")
            predicted = pd.Series(only_class, index=encoded.index, name="family", dtype=int)
            importance = pd.Series(0.0, index=encoded.columns, dtype=float).sort_values(ascending=False)
            return predicted, probs, confidence, None, importance

        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        idx_to_class = {idx: label for label, idx in class_to_idx.items()}

        y_train = labels.loc[labelled_mask].map(class_to_idx).astype(int)
        model = self._build_xgb_classifier(n_classes=len(classes))
        model.fit(encoded.loc[labelled_mask], y_train)

        prob_array = model.predict_proba(encoded)
        if prob_array.ndim == 1:
            prob_array = np.column_stack([1.0 - prob_array, prob_array])

        probs = pd.DataFrame(
            prob_array,
            index=encoded.index,
            columns=[idx_to_class[i] for i in range(prob_array.shape[1])],
            dtype=float,
        )
        pred_idx = probs.values.argmax(axis=1)
        predicted = pd.Series(
            [idx_to_class[int(i)] for i in pred_idx],
            index=encoded.index,
            name="family",
            dtype=int,
        )
        confidence = probs.max(axis=1).rename("confidence")
        importance = pd.Series(model.feature_importances_, index=encoded.columns, dtype=float)
        importance = importance.sort_values(ascending=False)
        return predicted, probs, confidence, model, importance

    def _fit_synthetic_anomaly_detector(
        self,
        encoded: pd.DataFrame,
    ) -> tuple[Any, pd.Series]:
        """Fit a binary XGBoost model against synthetic negatives."""
        n_synthetic = max(len(encoded), int(np.ceil(len(encoded) * self.synthetic_ratio)))
        synthetic = self._sample_synthetic_negatives(encoded, n_synthetic)
        X_binary = pd.concat([encoded, synthetic], axis=0, ignore_index=True)
        y_binary = np.concatenate([np.ones(len(encoded), dtype=int), np.zeros(len(synthetic), dtype=int)])

        model = self._build_xgb_binary_classifier()
        model.fit(X_binary, y_binary)
        p_real = model.predict_proba(encoded)[:, 1]
        anomaly_score = pd.Series(1.0 - p_real, index=encoded.index, dtype=float)
        return model, anomaly_score

    def _sample_synthetic_negatives(self, encoded: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        """Uniformly sample synthetic negatives inside the observed feature bounds."""
        rng = np.random.default_rng(self.random_state)
        mins = encoded.min(axis=0).to_numpy(dtype=float)
        maxs = encoded.max(axis=0).to_numpy(dtype=float)
        span = maxs - mins
        maxs = np.where(span <= 1e-12, mins + 1.0, maxs)

        sampled = rng.uniform(low=mins, high=maxs, size=(n_rows, encoded.shape[1]))
        return pd.DataFrame(sampled, columns=encoded.columns)

    def _align_cluster_labels(
        self,
        confident_X: pd.DataFrame,
        confident_labels: pd.Series,
        uncertain_X: pd.DataFrame,
        uncertain_labels: pd.Series,
        fallback_labels: pd.Series,
    ) -> pd.Series:
        """Align uncertain local clusters to the global class ids."""
        if confident_X.empty or uncertain_X.empty:
            return fallback_labels.astype(int)

        global_centroids = {
            int(label): confident_X.loc[confident_labels == label].mean(axis=0).to_numpy(dtype=float)
            for label in sorted(confident_labels.unique())
            if label >= 0
        }
        if not global_centroids:
            return fallback_labels.astype(int)

        mapping = {}
        for local_label in sorted(int(c) for c in uncertain_labels.unique() if c >= 0):
            local_centroid = uncertain_X.loc[uncertain_labels == local_label].mean(axis=0).to_numpy(dtype=float)
            nearest_global = min(
                global_centroids,
                key=lambda global_label: float(np.linalg.norm(local_centroid - global_centroids[global_label])),
            )
            mapping[local_label] = nearest_global

        aligned = fallback_labels.astype(int).copy()
        for idx, local_label in uncertain_labels.items():
            if int(local_label) in mapping:
                aligned.loc[idx] = mapping[int(local_label)]
        return aligned

    def _build_xgb_classifier(self, n_classes: int):
        """Construct a multiclass XGBoost classifier."""
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for pseudo-label refinement. Install it with `pip install xgboost`."
            ) from exc

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "tree_method": "hist",
            "verbosity": 0,
            "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
            "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
        }
        if n_classes > 2:
            params["num_class"] = n_classes
        return XGBClassifier(**params)

    def _build_xgb_binary_classifier(self):
        """Construct a binary XGBoost classifier."""
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for pseudo-label refinement. Install it with `pip install xgboost`."
            ) from exc

        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            verbosity=0,
            objective="binary:logistic",
            eval_metric="logloss",
        )

    def _base_probability_frame(self, labels: pd.Series) -> pd.DataFrame:
        """Convert hard labels into a degenerate probability matrix."""
        valid_classes = sorted(int(c) for c in labels.unique() if c >= 0)
        if not valid_classes:
            return pd.DataFrame(index=labels.index, dtype=float)

        probs = pd.DataFrame(0.0, index=labels.index, columns=valid_classes, dtype=float)
        for label in valid_classes:
            probs.loc[labels == label, label] = 1.0
        return probs
