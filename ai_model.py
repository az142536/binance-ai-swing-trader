"""
AI 趨勢波段預測模型 (AI Trend Swing Classifier)
基於向量化漸進提升樹 (GBDT) 與機率校準 (Z-Score Sigmoid Calibration)，
具備防未來函數的時間序列劃分、樣本平衡權重與特徵重要性分析。
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from features import FEATURE_NAMES

class PureNumpyEnsembleClassifier:
    """
    向量化漸進提升分類器 (Pure-NumPy Gradient Boosted Decision Ensemble)
    自動適配樣本不平衡，計算特徵重要性，並以 Newton-Raphson 步長更新葉節點。
    """
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.10, max_depth: int = 2, l2_reg: float = 1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_reg = l2_reg
        self.trees: List[Dict[str, Any]] = []
        self.bias: float = 0.0
        self.feature_importances_: np.ndarray = np.zeros(len(FEATURE_NAMES))

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))

    def _build_tree_node(self, X: np.ndarray, residuals: np.ndarray, hessians: np.ndarray, depth: int) -> Dict[str, Any]:
        """遞迴建立決策樹節點 (支援深度 1 或 2)"""
        n_samples, n_features = X.shape
        sum_grad = np.sum(residuals)
        sum_hess = np.sum(hessians)

        # 葉節點數值 (Newton 步長)
        leaf_value = float(sum_grad / (sum_hess + self.l2_reg))

        if depth >= self.max_depth or n_samples < 10:
            return {"is_leaf": True, "value": leaf_value}

        best_feat = -1
        best_thresh = 0.0
        best_gain = -1.0
        best_left_idx = None
        best_right_idx = None

        # 隨機子特徵抽樣
        cand_feats = np.random.choice(n_features, size=min(n_features, 12), replace=False)

        current_loss_score = (sum_grad ** 2) / (sum_hess + self.l2_reg)

        for feat in cand_feats:
            col = X[:, feat]
            # 依特徵分位數選取 7 個切分點候選
            thresholds = np.percentile(col, [15, 25, 40, 50, 60, 75, 85])
            for th in thresholds:
                left_mask = col <= th
                right_mask = ~left_mask
                n_l = np.sum(left_mask)
                n_r = np.sum(right_mask)
                if n_l < 5 or n_r < 5:
                    continue

                g_l = np.sum(residuals[left_mask])
                h_l = np.sum(hessians[left_mask])
                g_r = np.sum(residuals[right_mask])
                h_r = np.sum(hessians[right_mask])

                gain = 0.5 * ((g_l ** 2) / (h_l + self.l2_reg) + (g_r ** 2) / (h_r + self.l2_reg) - current_loss_score)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = float(th)
                    best_left_idx = left_mask
                    best_right_idx = right_mask

        if best_gain > 0.05 and best_left_idx is not None and best_right_idx is not None:
            self.feature_importances_[best_feat] += best_gain
            left_child = self._build_tree_node(X[best_left_idx], residuals[best_left_idx], hessians[best_left_idx], depth + 1)
            right_child = self._build_tree_node(X[best_right_idx], residuals[best_right_idx], hessians[best_right_idx], depth + 1)
            return {
                "is_leaf": False,
                "feat": int(best_feat),
                "thresh": float(best_thresh),
                "left": left_child,
                "right": right_child
            }

        return {"is_leaf": True, "value": leaf_value}

    def _predict_node(self, node: Dict[str, Any], row: np.ndarray) -> float:
        if node["is_leaf"]:
            return node["value"]
        if row[node["feat"]] <= node["thresh"]:
            return self._predict_node(node["left"], row)
        else:
            return self._predict_node(node["right"], row)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features, dtype=float)
        self.trees = []

        # 樣本加權：多頭波段為少數類，提高其重要性權重
        pos_count = max(np.sum(y == 1), 1)
        neg_count = max(np.sum(y == 0), 1)
        pos_weight = neg_count / pos_count
        weights = np.where(y == 1, pos_weight, 1.0)

        # 初始偏置
        p_init = np.clip(np.sum(y * weights) / np.sum(weights), 0.05, 0.95)
        self.bias = float(np.log(p_init / (1.0 - p_init)))

        f = np.full(n_samples, self.bias, dtype=float)

        for _ in range(self.n_estimators):
            p = self._sigmoid(f)
            # 一階與二階導數
            residuals = weights * (y - p)
            hessians = weights * (p * (1.0 - p))

            tree = self._build_tree_node(X, residuals, hessians, depth=0)
            self.trees.append(tree)

            # 更新 f
            for i in range(n_samples):
                val = self._predict_node(tree, X[i])
                f[i] += self.learning_rate * val

        tot = np.sum(self.feature_importances_)
        if tot > 0:
            self.feature_importances_ /= tot

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        f = np.full(n_samples, self.bias, dtype=float)
        for i in range(n_samples):
            row = X[i]
            for tree in self.trees:
                f[i] += self.learning_rate * self._predict_node(tree, row)
        return f

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        p1 = self._sigmoid(logits)
        return np.column_stack([1.0 - p1, p1])


class AISwingModel:
    """高階 AI 趨勢波段模型封裝器 (含標準化與機率校準)"""
    def __init__(self):
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.model: Optional[PureNumpyEnsembleClassifier] = None
        self.feature_names = FEATURE_NAMES
        self.trained_metrics: Dict[str, Any] = {}
        
        # 校準參數
        self.calib_mean: float = 0.0
        self.calib_std: float = 1.0

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.scaler_mean = np.nanmean(X, axis=0)
            self.scaler_std = np.nanstd(X, axis=0)
            self.scaler_std[self.scaler_std == 0] = 1.0

        X_clean = np.where(np.isnan(X), self.scaler_mean if self.scaler_mean is not None else 0.0, X)
        return (X_clean - self.scaler_mean) / self.scaler_std

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))

    def train(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Dict[str, Any]:
        """
        時序防未來函數劃分訓練模型 (80% 歷史訓練集，20% 未來驗證集)
        """
        n_samples = len(y)
        split_idx = int(n_samples * train_ratio)

        X_train_raw, y_train = X[:split_idx], y[:split_idx]
        X_val_raw, y_val = X[split_idx:], y[split_idx:]

        X_train = self._normalize(X_train_raw, fit=True)
        X_val = self._normalize(X_val_raw, fit=False)

        self.model = PureNumpyEnsembleClassifier(n_estimators=45, learning_rate=0.10, max_depth=2)
        self.model.fit(X_train, y_train)

        # 訓練集校準 (Z-Score Calibration): 將輸出 logit 校準為 0~1 的交易信心指數
        train_logits = self.model.predict_logits(X_train)
        self.calib_mean = float(np.mean(train_logits))
        self.calib_std = float(max(np.std(train_logits), 1e-4))

        # 驗證集校準後機率
        val_logits = self.model.predict_logits(X_val)
        val_calib_z = (val_logits - self.calib_mean) / self.calib_std
        val_probs = self._sigmoid(val_calib_z * 1.2) # 放大動態範圍

        threshold = 0.58
        val_preds = (val_probs >= threshold).astype(int)

        accuracy = float(np.mean(val_preds == y_val))
        tp = int(np.sum((val_preds == 1) & (y_val == 1)))
        fp = int(np.sum((val_preds == 1) & (y_val == 0)))
        fn = int(np.sum((val_preds == 0) & (y_val == 1)))
        tn = int(np.sum((val_preds == 0) & (y_val == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        self.trained_metrics = {
            "total_samples": n_samples,
            "train_samples": split_idx,
            "val_samples": n_samples - split_idx,
            "positive_wave_ratio": float(np.mean(y_train)),
            "val_accuracy": round(accuracy * 100, 2),
            "val_precision": round(precision * 100, 2),
            "val_recall": round(recall * 100, 2),
            "val_f1": round(f1, 4),
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        }

        return self.trained_metrics

    def predict_bullish_prob(self, X: np.ndarray) -> np.ndarray:
        """
        預測 K 線出現多頭趨勢波段的信心指數 (0.0 ~ 1.0)
        - 0.50: 市場處於平均/平淡狀態
        - >= 0.60: 多頭波段特徵明顯成形，具備開倉條件
        - <= 0.35: 趨勢動能衰竭或轉為空頭
        """
        if self.model is None or self.scaler_mean is None:
            raise RuntimeError("模型尚未訓練或載入！請先執行 train 指令。")
        X_norm = self._normalize(X, fit=False)
        logits = self.model.predict_logits(X_norm)
        calib_z = (logits - self.calib_mean) / self.calib_std
        probs = self._sigmoid(calib_z * 1.2)
        return probs

    def get_feature_importances(self) -> Dict[str, float]:
        """獲取各特徵重要性排名"""
        if self.model is None:
            return {}
        imp = self.model.feature_importances_
        sorted_indices = np.argsort(imp)[::-1]
        return {self.feature_names[i]: round(float(imp[i] * 100), 2) for i in sorted_indices}

    def save(self, file_path: Path):
        """保存模型結構與校準參數"""
        data = {
            "scaler_mean": self.scaler_mean.tolist() if self.scaler_mean is not None else [],
            "scaler_std": self.scaler_std.tolist() if self.scaler_std is not None else [],
            "calib_mean": self.calib_mean,
            "calib_std": self.calib_std,
            "bias": self.model.bias if self.model else 0.0,
            "trees": self.model.trees if self.model else [],
            "feature_importances": self.model.feature_importances_.tolist() if self.model else [],
            "metrics": self.trained_metrics
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, file_path: Path):
        """載入保存的模型"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.scaler_mean = np.array(data["scaler_mean"], dtype=float)
        self.scaler_std = np.array(data["scaler_std"], dtype=float)
        self.calib_mean = float(data.get("calib_mean", 0.0))
        self.calib_std = float(data.get("calib_std", 1.0))
        self.model = PureNumpyEnsembleClassifier()
        self.model.bias = float(data["bias"])
        self.model.trees = data["trees"]
        self.model.feature_importances_ = np.array(data["feature_importances"], dtype=float)
        self.trained_metrics = data.get("metrics", {})
