"""
Model Training Module for 4D Lottery Prediction
Implements multiple approaches: Statistical, XGBoost, LSTM
"""

import pandas as pd
import numpy as np
from collections import Counter
import pickle
import os
import random
from importlib.util import find_spec

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Statistical and Markov methods must not import optional ML runtimes.
HAS_XGBOOST = find_spec('xgboost') is not None
HAS_TENSORFLOW = find_spec('tensorflow') is not None


class StatisticalModel:
    """Simple statistical model based on frequency analysis."""
    
    def __init__(self):
        self.number_freq = {}
        self.digit_freq = [{} for _ in range(4)]
        self.permutation_freq = {}
        self.name = "Statistical"
    
    def train(self, df: pd.DataFrame):
        """Train on historical data."""
        self.number_freq = Counter()
        self.digit_freq = [Counter() for _ in range(4)]
        self.permutation_freq = Counter()
        
        for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
            if col in df.columns:
                for num in df[col].astype(str).str.zfill(4):
                    self.number_freq[num] += 1
                    for i, digit in enumerate(num):
                        self.digit_freq[i][digit] += 1
                    # Permutation (sorted digits)
                    perm = ''.join(sorted(num))
                    self.permutation_freq[perm] += 1
        
        print(f"Statistical model trained on {len(df)} draws")
        print(f"  Unique numbers seen: {len(self.number_freq)}")
    
    def predict(self, n: int = 20) -> list:
        """Predict top N numbers based on historical frequency."""
        # Method 1: Most frequent numbers
        freq_based = [num for num, _ in self.number_freq.most_common(n)]
        if len(freq_based) >= n:
            return freq_based
        
        # Method 2: Generate from digit frequencies
        digit_based = []
        for _ in range(n):
            num = ''
            for i in range(4):
                # Weighted random choice based on digit frequency
                digits, weights = zip(*self.digit_freq[i].items())
                weights = np.array(weights, dtype=float)
                weights /= weights.sum()
                num += np.random.choice(digits, p=weights)
            digit_based.append(num)
        
        # Method 3: Most frequent permutation families
        perm_based = []
        for perm, _ in self.permutation_freq.most_common(n * 2):
            # Generate a random arrangement of this permutation
            import random
            perm_list = list(perm)
            random.shuffle(perm_list)
            perm_based.append(''.join(perm_list))
            if len(perm_based) >= n:
                break
        
        # Combine all methods with deduplication
        all_predictions = []
        seen = set()
        for num in freq_based + digit_based + perm_based:
            if num not in seen:
                all_predictions.append(num)
                seen.add(num)
            if len(all_predictions) >= n:
                break
        
        return all_predictions[:n]
    
    def get_number_score(self, number: str) -> float:
        """Get a score for a specific number."""
        number = str(number).zfill(4)
        
        # Frequency score
        freq_score = self.number_freq.get(number, 0)
        
        # Digit position score
        digit_score = sum(
            self.digit_freq[i].get(digit, 0) 
            for i, digit in enumerate(number)
        )
        
        # Permutation score
        perm = ''.join(sorted(number))
        perm_score = self.permutation_freq.get(perm, 0)
        
        # Combined score (weighted)
        total = freq_score * 3 + digit_score * 0.1 + perm_score * 1
        return total
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'number_freq': self.number_freq,
                'digit_freq': self.digit_freq,
                'permutation_freq': self.permutation_freq
            }, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.number_freq = data['number_freq']
            self.digit_freq = data['digit_freq']
            self.permutation_freq = data['permutation_freq']


class MarkovChainModel:
    """Markov chain model for digit transitions."""

    def __init__(self):
        self.transition_matrix = {}
        self.initial_probs = {}
        self.name = "MarkovChain"

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'transition_matrix': self.transition_matrix,
                'initial_probs': self.initial_probs,
            }, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.transition_matrix = data['transition_matrix']
            self.initial_probs = data['initial_probs']

    def get_number_score(self, number: str) -> float:
        """Likelihood under chain: P(d0) * prod P(d_i | d_{i-1})."""
        number = str(number).zfill(4)
        if not self.initial_probs:
            return 0.0
        p = self.initial_probs.get(number[0], 0.0)
        for i in range(1, 4):
            row = self.transition_matrix.get(number[i - 1], {})
            p *= row.get(number[i], 0.0)
        return p
    
    def train(self, df: pd.DataFrame):
        """Train transition probabilities."""
        # Count digit transitions within numbers
        transitions = Counter()
        initials = Counter()
        
        for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
            if col in df.columns:
                for num in df[col].astype(str).str.zfill(4):
                    initials[num[0]] += 1
                    for i in range(len(num) - 1):
                        transitions[(num[i], num[i+1])] += 1
        
        # Convert to probabilities
        self.initial_probs = {}
        total_init = sum(initials.values())
        for digit, count in initials.items():
            self.initial_probs[digit] = count / total_init
        
        self.transition_matrix = {}
        for digit in '0123456789':
            total = sum(transitions.get((digit, next_d), 0) for next_d in '0123456789')
            if total > 0:
                self.transition_matrix[digit] = {
                    next_d: transitions.get((digit, next_d), 0) / total
                    for next_d in '0123456789'
                }
            else:
                self.transition_matrix[digit] = {d: 0.1 for d in '0123456789'}
        
        print(f"Markov Chain model trained on {len(df)} draws")
    
    def predict(self, n: int = 20) -> list:
        """Generate numbers using the Markov chain."""
        predictions = []
        
        for _ in range(n * 2):  # Generate more, then deduplicate
            digits, probs = zip(*self.initial_probs.items())
            probs = np.array(probs)
            probs /= probs.sum()
            num = np.random.choice(digits, p=probs)
            
            for _ in range(3):
                last_digit = num[-1]
                next_digits = list(self.transition_matrix[last_digit].keys())
                next_probs = np.array(list(self.transition_matrix[last_digit].values()))
                next_probs /= next_probs.sum()
                num += np.random.choice(next_digits, p=next_probs)
            
            if num not in predictions:
                predictions.append(num)
            
            if len(predictions) >= n:
                break
        
        return predictions[:n]


class XGBoostModel:
    """Historical appeared-vs-never diagnostic, not a next-draw classifier.

    The frequency feature determines this historical label. Scores must not
    be interpreted as future winning probabilities.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.name = "XGBoost"
    
    def _create_features(self, number: str, df: pd.DataFrame) -> dict:
        """Create feature vector for a number."""
        number = str(number).zfill(4)
        digits = [int(d) for d in number]
        
        # Count historical appearances
        freq = 0
        for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
            if col in df.columns:
                freq += (df[col].astype(str).str.zfill(4) == number).sum()
        
        features = {
            'd0': digits[0], 'd1': digits[1], 'd2': digits[2], 'd3': digits[3],
            'digit_sum': sum(digits),
            'digit_mean': np.mean(digits),
            'digit_std': np.std(digits),
            'odd_count': sum(1 for d in digits if d % 2 == 1),
            'unique_digits': len(set(digits)),
            'has_zero': 1 if 0 in digits else 0,
            'freq': freq,
        }
        
        return features
    
    def train(self, df: pd.DataFrame):
        """Train XGBoost on historical data."""
        if not HAS_XGBOOST:
            print("XGBoost not available!")
            return
        import xgboost as xgb
        
        # Create training data
        # Positive samples: numbers that actually appeared
        positive_numbers = set()
        for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
            if col in df.columns:
                positive_numbers.update(df[col].astype(str).str.zfill(4))
        
        # Negative samples: random numbers that never appeared
        all_numbers = set(f'{i:04d}' for i in range(10000))
        negative_numbers = list(all_numbers - positive_numbers)
        np.random.shuffle(negative_numbers)
        negative_numbers = negative_numbers[:len(positive_numbers)]  # Balance
        
        # Create feature matrix
        X_data = []
        y_data = []
        
        for num in positive_numbers:
            X_data.append(self._create_features(num, df))
            y_data.append(1)
        
        for num in negative_numbers:
            X_data.append(self._create_features(num, df))
            y_data.append(0)
        
        X = pd.DataFrame(X_data)
        y = np.array(y_data)
        
        self.feature_names = X.columns.tolist()
        
        # Train model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X, y)
        
        print(f"XGBoost model trained on {len(X)} samples ({len(positive_numbers)} positive)")
    
    def predict(self, n: int = 20, df: pd.DataFrame = None) -> list:
        """Predict top N likely numbers."""
        if self.model is None:
            print("Model not trained!")
            return []
        
        # Score all possible numbers
        all_numbers = [f'{i:04d}' for i in range(10000)]
        X_data = [self._create_features(num, df) for num in all_numbers]
        X = pd.DataFrame(X_data)
        
        probs = self.model.predict_proba(X)[:, 1]
        
        # Get top N
        top_indices = np.argsort(probs)[-n:][::-1]
        predictions = [(all_numbers[i], probs[i]) for i in top_indices]
        
        return predictions
    
    def save(self, filepath: str):
        if self.model:
            self.model.save_model(filepath)
    
    def load(self, filepath: str):
        if HAS_XGBOOST:
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(filepath)


class LSTMModel:
    """Real LSTM over per-digit one-hot stream. Predicts next-digit distribution."""

    def __init__(self, sequence_length: int = 40):
        # sequence_length = number of past digits fed (10 draws * 4 digits = 40)
        self.model = None
        self.sequence_length = sequence_length
        self.name = "LSTM"

    def _digit_stream(self, df: pd.DataFrame, prize_col: str = '1stPrizeNo') -> np.ndarray:
        df = df.sort_values('DrawDate', kind='stable') if 'DrawDate' in df.columns else df
        numbers = df[prize_col].astype(str).str.zfill(4).values
        digits = np.fromiter(
            (int(c) for num in numbers for c in num),
            dtype=np.int8,
            count=len(numbers) * 4,
        )
        return digits

    def _prepare_sequences(self, df: pd.DataFrame, prize_col: str = '1stPrizeNo'):
        stream = self._digit_stream(df, prize_col)
        L = self.sequence_length
        if len(stream) <= L:
            return np.empty((0, L, 10)), np.empty((0,), dtype=np.int64)

        # one-hot encode whole stream once
        onehot = np.eye(10, dtype=np.float32)[stream]

        n = len(stream) - L
        X = np.lib.stride_tricks.sliding_window_view(onehot, (L, 10))[:n].reshape(n, L, 10)
        y = stream[L:L + n].astype(np.int64)
        return X, y

    def train(self, df: pd.DataFrame, epochs: int = 5, batch_size: int = 128):
        if not HAS_TENSORFLOW:
            print("TensorFlow not available!")
            return
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        X, y = self._prepare_sequences(df)
        if len(X) == 0:
            print("Not enough data for LSTM training!")
            return

        tf.random.set_seed(SEED)
        self.model = Sequential([
            tf.keras.Input(shape=(self.sequence_length, 10)),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax'),
        ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        print(f"Training LSTM on {len(X)} sequences ({epochs} epochs)...")
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
        )

    def _next_digit_probs(self, seed_digits: np.ndarray) -> np.ndarray:
        L = self.sequence_length
        window = seed_digits[-L:]
        if len(window) < L:
            window = np.concatenate([np.zeros(L - len(window), dtype=np.int8), window])
        X = np.eye(10, dtype=np.float32)[window][np.newaxis, ...]
        return self.model.predict(X, verbose=0)[0]

    def predict(self, df: pd.DataFrame, n: int = 20, prize_col: str = '1stPrizeNo') -> list:
        if self.model is None:
            print("Model not trained!")
            return []

        seed = self._digit_stream(df.tail(max(50, self.sequence_length // 4 + 5)), prize_col)
        rng = np.random.default_rng(SEED)

        predictions = []
        seen = set()
        attempts = 0
        while len(predictions) < n and attempts < n * 10:
            attempts += 1
            digits = list(seed[-self.sequence_length:])
            num_digits = []
            for _ in range(4):
                probs = self._next_digit_probs(np.array(digits, dtype=np.int8))
                d = int(rng.choice(10, p=probs))
                num_digits.append(d)
                digits.append(d)
            num = ''.join(str(d) for d in num_digits)
            if num not in seen:
                seen.add(num)
                predictions.append(num)

        return predictions

    def get_number_score(self, df: pd.DataFrame, number: str, prize_col: str = '1stPrizeNo') -> float:
        """Score = P(number) under LSTM given last seen window."""
        if self.model is None:
            return 0.0
        number = str(number).zfill(4)
        seed = self._digit_stream(df.tail(max(50, self.sequence_length // 4 + 5)), prize_col)
        digits = list(seed[-self.sequence_length:])
        p = 1.0
        for c in number:
            probs = self._next_digit_probs(np.array(digits, dtype=np.int8))
            p *= float(probs[int(c)])
            digits.append(int(c))
        return p

    def save(self, filepath: str):
        if self.model is not None:
            self.model.save(filepath)

    def load(self, filepath: str):
        if HAS_TENSORFLOW and os.path.exists(filepath):
            import tensorflow as tf
            self.model = tf.keras.models.load_model(filepath)


def train_all_models(df: pd.DataFrame) -> dict:
    """Train all available models."""
    models = {}
    
    # Statistical model (always available)
    print("\n=== Training Statistical Model ===")
    stat_model = StatisticalModel()
    stat_model.train(df)
    models['statistical'] = stat_model
    
    # Markov Chain model
    print("\n=== Training Markov Chain Model ===")
    markov_model = MarkovChainModel()
    markov_model.train(df)
    models['markov'] = markov_model
    
    # XGBoost model
    if HAS_XGBOOST:
        print("\n=== Training XGBoost Model ===")
        xgb_model = XGBoostModel()
        xgb_model.train(df)
        models['xgboost'] = xgb_model
    
    # LSTM model
    if HAS_TENSORFLOW:
        print("\n=== Training LSTM Model ===")
        lstm_model = LSTMModel(sequence_length=40)
        lstm_model.train(df, epochs=5)
        models['lstm'] = lstm_model
    
    return models


if __name__ == "__main__":
    from data_loader import load_combined_data, extract_top3_prizes
    
    # Load data
    df = load_combined_data()
    top3 = extract_top3_prizes(df)
    
    # Train models
    models = train_all_models(top3)
    
    # Test predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS FROM EACH MODEL")
    print("="*50)
    
    for name, model in models.items():
        print(f"\n{model.name} Model Predictions:")
        if name == 'xgboost':
            preds = model.predict(n=10, df=top3)
            for num, prob in preds:
                print(f"  {num} (score: {prob:.4f})")
        elif name == 'lstm':
            preds = model.predict(df=top3, n=10)
            print(f"  {preds}")
        else:
            preds = model.predict(n=10)
            print(f"  {preds}")
