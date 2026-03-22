"""
hmm_classifier.py
-----------------
Trains a Gaussian HMM on the 5 regime features.
Outputs: regime label (0-3) + posterior probabilities for every day.

Key idea: regimes are HIDDEN. We observe features, infer the state.
The HMM also captures regime STICKINESS — markets don't jump between
states every day. It learns that bull markets tend to persist.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

log = logging.getLogger(__name__)

# Where to save the trained model
MODEL_PATH = Path(__file__).parent.parent.parent / "logs" / "hmm_model.pkl"


class RegimeClassifier:
    """
    Wraps hmmlearn's GaussianHMM with:
      - clean fit / predict interface
      - automatic regime labeling (which state = bull, crisis, etc.)
      - save / load so you don't retrain every run
    """

    def __init__(self, n_states: int = 4,
                 n_iter: int = 200,
                 random_state: int = 42):
        """
        n_states     : number of regimes to find (we use 4)
        n_iter       : how many EM iterations to run (more = more accurate,
                       but 200 is enough — beyond that it barely improves)
        random_state : fix the seed so results are reproducible
        covariance_type="full" means each regime gets its own full
        covariance matrix — captures correlations between features
        within each regime, not just individual feature variances.
        """
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            verbose=False,
        )
        self.regime_map = {}      # maps HMM state int → label string
        self.is_fitted  = False
        self.feature_cols = None  # remember which columns we trained on

    # ── TRAINING ────────────────────────────────────────────────────

    def fit(self, features_norm: pd.DataFrame) -> "RegimeClassifier":
        """
        Train the HMM on normalized feature matrix.

        The HMM uses the Baum-Welch algorithm (a type of EM) internally.
        It iterates between:
          E-step: given current params, estimate which state each day is in
          M-step: given state estimates, update the Gaussian params per state
        This repeats until convergence (or n_iter reached).

        Parameters
        ----------
        features_norm : pd.DataFrame — normalized features from Stage 2
                        (mean=0, std=1 per column)
        """
        self.feature_cols = list(features_norm.columns)
        X = features_norm.values  # HMM needs a numpy array, not DataFrame

        log.info(f"Training HMM: {self.n_states} states, "
                 f"{len(X)} observations, {X.shape[1]} features ...")

        self.model.fit(X)
        self.is_fitted = True

        log.info("HMM training complete.")
        log.info(f"  Converged: {self.model.monitor_.converged}")
        log.info(f"  Log-likelihood: {self.model.score(X):.2f}")

        # After fitting, auto-label the regimes by inspecting learned means
        self._label_regimes(features_norm)

        return self  # allows chaining: classifier.fit(X).predict(X)

    def _label_regimes(self, features_norm: pd.DataFrame):
        """
        The HMM numbers states 0, 1, 2, 3 — but which is the bull regime?
        We figure it out by inspecting the learned mean vector for each state.

        Strategy:
          - State with HIGHEST vol mean     → crisis or high-vol trend
          - State with LOWEST vol mean      → bull or choppy (split by Sharpe)
          - State with most negative Sharpe → crisis
          - State with most positive Sharpe → bull

        This gives us a human-readable label for each HMM state number.
        """
        means = pd.DataFrame(
            self.model.means_,
            columns=self.feature_cols
        )
        log.info("\nLearned feature means per state (normalized):")
        log.info(means.round(3).to_string())

        vol_col    = "realized_vol"
        sharpe_col = "rolling_sharpe"
        autocorr   = "autocorr_5d"

        labels = {}
        vol_means    = means[vol_col]
        sharpe_means = means[sharpe_col]

        # Crisis = highest vol AND most negative Sharpe
        crisis_state = (vol_means - sharpe_means).idxmax()
        labels[crisis_state] = "crisis"

        # Bull = lowest vol AND highest Sharpe
        bull_state = (sharpe_means - vol_means).idxmax()
        labels[bull_state] = "bull"

        # Remaining two: split by autocorrelation
        remaining = [s for s in range(self.n_states)
                     if s not in labels]
        for s in remaining:
            if means.loc[s, autocorr] >= 0:
                labels[s] = "high_vol_trend"
            else:
                labels[s] = "choppy"

        self.regime_map = labels
        log.info(f"\nRegime map: {labels}")

    # ── PREDICTION ──────────────────────────────────────────────────

    def predict(self, features_norm: pd.DataFrame) -> pd.DataFrame:
        """
        For each day in features_norm, return:
          - regime_id    : integer HMM state (0-3)
          - regime_label : human-readable string (bull, choppy, etc.)
          - prob_bull, prob_choppy, prob_high_vol_trend, prob_crisis
            : posterior probability of being in each regime

        The probabilities are what the meta-allocator will use in Stage 4.
        """
        assert self.is_fitted, "Call .fit() before .predict()"
        X = features_norm.values

        # Hard assignment: most likely state per day
        regime_ids = self.model.predict(X)

        # Soft assignment: probability distribution over states per day
        # Shape: (n_days, n_states)
        posteriors = self.model.predict_proba(X)

        # Build output DataFrame
        result = pd.DataFrame(index=features_norm.index)
        result["regime_id"]    = regime_ids
        result["regime_label"] = [self.regime_map.get(r, f"state_{r}")
                                   for r in regime_ids]

        # Add a probability column for each named regime
        inv_map = {v: k for k, v in self.regime_map.items()}
        for name in ["bull", "choppy", "high_vol_trend", "crisis"]:
            state_id = inv_map.get(name)
            if state_id is not None:
                result[f"prob_{name}"] = posteriors[:, state_id]
            else:
                result[f"prob_{name}"] = 0.0

        return result

    def transition_matrix(self) -> pd.DataFrame:
        """
        Return the learned transition probability matrix as a DataFrame.
        Row = current state, Column = next state.
        Diagonal should be high (regimes are sticky).
        """
        assert self.is_fitted
        labels = [self.regime_map.get(i, f"state_{i}")
                  for i in range(self.n_states)]
        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels
        )

    # ── SAVE / LOAD ─────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH):
        """Save the trained classifier using joblib (pickle-safe)."""
        import joblib
        path.parent.mkdir(exist_ok=True)
        joblib.dump(self, path)
        log.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "RegimeClassifier":
        """Load a previously saved classifier using joblib."""
        import joblib
        obj = joblib.load(path)
        log.info(f"Model loaded ← {path}")
        return obj

# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from regimesense.data.fetcher import load_data_from_config
    from regimesense.features.regime_features import (
        build_feature_matrix, normalize_features
    )

    # Load data + build features
    df       = load_data_from_config()
    features = build_feature_matrix(df)
    normed   = normalize_features(features)

    # Train
    clf = RegimeClassifier(n_states=4, n_iter=200, random_state=42)
    clf.fit(normed)

    # Predict
    regimes = clf.predict(normed)

    # Show results
    print("\nRegime distribution (% of days in each regime):")
    pct = regimes["regime_label"].value_counts(normalize=True) * 100
    print(pct.round(1).to_string())

    print("\nTransition matrix (rows=from, cols=to):")
    print(clf.transition_matrix().round(3).to_string())

    print("\nSample output (last 5 rows):")
    print(regimes.tail().round(3).to_string())

    # Save model
    clf.save()
    print("\nModel saved to logs/hmm_model.pkl")