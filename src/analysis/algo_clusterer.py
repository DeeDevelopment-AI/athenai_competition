"""
Clustering de algoritmos usando múltiples técnicas.

Traditional Methods:
- GMM: Gaussian Mixture Model (clusters probabilísticos)
- KMEANS: K-Means clásico (clusters esféricos)
- HIERARCHICAL: Agglomerative clustering (dendrograma)
- DBSCAN: Density-based (detecta outliers, no requiere k)
- HDBSCAN: Hierarchical DBSCAN (mejor manejo de densidades variables)

Deep Learning Methods:
- AUTOENCODER: Basic autoencoder + K-means on latent space
- VAE: Variational Autoencoder (probabilistic latent space)
- SPARSE_AE: Sparse Autoencoder (sparsity regularization)
- DEEP_INFOMAX: Deep InfoMax (mutual information maximization)
- BIGAN: Bidirectional GAN (adversarial representation learning)
- IIC: Invariant Information Clustering (direct clustering optimization)
- DAC: Deep Adaptive Clustering (pairwise similarity learning)

Enfoque de dos capas:
1. Capa 1 (Life Profile): Agrupa por patrón de actividad
2. Capa 2 (Financial Behavior): Dentro de cada grupo, agrupa por rendimiento
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

logger = logging.getLogger(__name__)


class ClusterMethod(Enum):
    """Métodos de clustering disponibles."""
    # Traditional methods
    GMM = "gmm"
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    # Deep learning methods
    AUTOENCODER = "autoencoder"          # Basic autoencoder + K-means
    VAE = "vae"                          # Variational Autoencoder
    SPARSE_AE = "sparse_ae"              # Sparse Autoencoder
    DEEP_INFOMAX = "deep_infomax"        # Deep InfoMax (mutual information)
    BIGAN = "bigan"                      # Bidirectional GAN
    IIC = "iic"                          # Invariant Information Clustering
    DAC = "dac"                          # Deep Adaptive Clustering


class ScalerType(Enum):
    """Tipos de escalado disponibles."""
    STANDARD = "standard"
    ROBUST = "robust"  # Mejor para datos con outliers


@dataclass
class ClusterResult:
    """Resultado de un clustering."""
    method: ClusterMethod
    labels: np.ndarray
    n_clusters: int
    n_noise: int  # Puntos marcados como ruido (DBSCAN/HDBSCAN)
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    cluster_sizes: dict[int, int]
    model: object  # Modelo entrenado (para predicciones)
    features_used: list[str] = field(default_factory=list)


@dataclass
class TwoLayerClusterResult:
    """Resultado del clustering de dos capas."""
    # Capa 1: Life Profile
    life_profile_result: ClusterResult
    life_profile_names: dict[int, str]

    # Capa 2: Financial Behavior (por cada life profile)
    behavior_results: dict[int, ClusterResult]  # {life_profile_id: ClusterResult}
    behavior_names: dict[int, dict[int, str]]  # {life_profile_id: {cluster_id: name}}

    # Labels combinados
    combined_labels: pd.Series  # "life_profile_behavior" format

    # Métricas agregadas
    overall_silhouette: float
    n_total_clusters: int


class AlgoClusterer:
    """
    Clustering de algoritmos con múltiples métodos.

    Uso básico:
        clusterer = AlgoClusterer(method=ClusterMethod.GMM, n_clusters=5)
        result = clusterer.fit(features_df)
        labels = result.labels

    Uso con dos capas:
        result = AlgoClusterer.two_layer_clustering(
            features_df,
            life_features=['start_idx', 'duration_ratio', 'active_ratio'],
            behavior_features=['sharpe', 'max_dd', 'return_decay']
        )
    """

    # Features por defecto para clustering simple
    DEFAULT_FEATURES = [
        'ann_return', 'ann_vol', 'sharpe', 'max_dd',
        'skewness', 'autocorr_1', 'trend_score', 'corr_benchmark'
    ]

    def __init__(
        self,
        method: ClusterMethod = ClusterMethod.GMM,
        n_clusters: int = 5,
        features: Optional[list[str]] = None,
        random_state: int = 42,
        scaler_type: ScalerType = ScalerType.ROBUST,
        # DBSCAN params
        eps: float = 0.5,
        min_samples: int = 5,
        # HDBSCAN params
        min_cluster_size: int = 15,
        # Autoencoder params
        ae_latent_dim: int = 8,
        ae_hidden_dims: tuple = (32, 16),
        ae_epochs: int = 100,
        ae_batch_size: int = 32,
        ae_learning_rate: float = 1e-3,
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.features = features or self.DEFAULT_FEATURES
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        # Autoencoder params
        self.ae_latent_dim = ae_latent_dim
        self.ae_hidden_dims = ae_hidden_dims
        self.ae_epochs = ae_epochs
        self.ae_batch_size = ae_batch_size
        self.ae_learning_rate = ae_learning_rate

        # Seleccionar scaler
        if scaler_type == ScalerType.ROBUST:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        self.model = None
        self._X_scaled = None
        self._feature_names = None

    def fit(self, algo_features: pd.DataFrame) -> ClusterResult:
        """
        Ajusta el clustering a los datos.

        Args:
            algo_features: DataFrame con features por algoritmo (index=algo_id)

        Returns:
            ClusterResult con labels y métricas
        """
        # Preparar datos
        X = self._prepare_data(algo_features)
        self._X_scaled = self.scaler.fit_transform(X)
        self._feature_names = list(X.columns)

        # Ejecutar clustering según método
        if self.method == ClusterMethod.GMM:
            labels = self._fit_gmm()
        elif self.method == ClusterMethod.KMEANS:
            labels = self._fit_kmeans()
        elif self.method == ClusterMethod.HIERARCHICAL:
            labels = self._fit_hierarchical()
        elif self.method == ClusterMethod.DBSCAN:
            labels = self._fit_dbscan()
        elif self.method == ClusterMethod.HDBSCAN:
            labels = self._fit_hdbscan()
        elif self.method == ClusterMethod.AUTOENCODER:
            labels = self._fit_autoencoder()
        elif self.method == ClusterMethod.VAE:
            labels = self._fit_vae()
        elif self.method == ClusterMethod.SPARSE_AE:
            labels = self._fit_sparse_autoencoder()
        elif self.method == ClusterMethod.DEEP_INFOMAX:
            labels = self._fit_deep_infomax()
        elif self.method == ClusterMethod.BIGAN:
            labels = self._fit_bigan()
        elif self.method == ClusterMethod.IIC:
            labels = self._fit_iic()
        elif self.method == ClusterMethod.DAC:
            labels = self._fit_dac()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Calcular métricas
        result = self._compute_metrics(labels)
        result.features_used = self._feature_names

        logger.info(
            f"{self.method.value}: {result.n_clusters} clusters, "
            f"silhouette={result.silhouette:.3f}, "
            f"noise={result.n_noise}"
        )

        return result

    def _prepare_data(self, algo_features: pd.DataFrame) -> pd.DataFrame:
        """Prepara y limpia datos para clustering."""
        available_features = [f for f in self.features if f in algo_features.columns]

        if len(available_features) < len(self.features):
            missing = set(self.features) - set(available_features)
            logger.warning(f"Missing features: {missing}")

        if len(available_features) == 0:
            raise ValueError(f"No features available. Requested: {self.features}")

        X = algo_features[available_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        return X

    def _fit_gmm(self) -> np.ndarray:
        """Gaussian Mixture Model clustering."""
        self.model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='full',
            random_state=self.random_state,
            n_init=3,
        )
        return self.model.fit_predict(self._X_scaled)

    def _fit_kmeans(self) -> np.ndarray:
        """K-Means clustering."""
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        return self.model.fit_predict(self._X_scaled)

    def _fit_hierarchical(self) -> np.ndarray:
        """Agglomerative Hierarchical clustering."""
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='ward',
        )
        return self.model.fit_predict(self._X_scaled)

    def _fit_dbscan(self) -> np.ndarray:
        """DBSCAN density-based clustering."""
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
        )
        return self.model.fit_predict(self._X_scaled)

    def _fit_hdbscan(self) -> np.ndarray:
        """HDBSCAN clustering."""
        try:
            import hdbscan
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
            )
            return self.model.fit_predict(self._X_scaled)
        except ImportError:
            logger.warning("hdbscan not installed, falling back to DBSCAN")
            return self._fit_dbscan()

    def _fit_autoencoder(self) -> np.ndarray:
        """
        Compression-based clustering using Autoencoder.

        Steps:
        1. Train autoencoder to compress features to latent space
        2. Extract latent representations
        3. Apply K-Means clustering on latent space
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.warning("PyTorch not installed, falling back to K-Means")
            return self._fit_kmeans()

        # Set random seed for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = self._X_scaled.shape[1]
        latent_dim = min(self.ae_latent_dim, input_dim - 1)

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in self.ae_hidden_dims:
            if hidden_dim >= prev_dim:
                hidden_dim = prev_dim // 2
            if hidden_dim < latent_dim:
                break
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        # Build decoder layers (mirror of encoder)
        decoder_layers = [nn.Linear(latent_dim, prev_dim), nn.ReLU()]
        for hidden_dim in reversed(self.ae_hidden_dims):
            if hidden_dim >= prev_dim or hidden_dim < latent_dim:
                continue
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        # Define autoencoder
        class Autoencoder(nn.Module):
            def __init__(self, encoder_layers, decoder_layers):
                super().__init__()
                self.encoder = nn.Sequential(*encoder_layers)
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x):
                z = self.encoder(x)
                x_recon = self.decoder(z)
                return x_recon, z

            def encode(self, x):
                return self.encoder(x)

        autoencoder = Autoencoder(encoder_layers, decoder_layers)

        # Training setup
        X_tensor = torch.FloatTensor(self._X_scaled)
        dataset = TensorDataset(X_tensor, X_tensor)

        batch_size = min(self.ae_batch_size, len(X_tensor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=self.ae_learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        autoencoder.train()
        for epoch in range(self.ae_epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                x_recon, _ = autoencoder(batch_x)
                loss = criterion(x_recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Extract latent representations
        autoencoder.eval()
        with torch.no_grad():
            latent_repr = autoencoder.encode(X_tensor).numpy()

        # Cluster in latent space using K-Means
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(latent_repr)

        # Store model for later use
        self.model = {
            'autoencoder': autoencoder,
            'kmeans': kmeans,
            'latent_dim': latent_dim,
        }

        logger.debug(f"Autoencoder: compressed {input_dim}D -> {latent_dim}D latent space")

        return labels

    def _fit_vae(self) -> np.ndarray:
        """
        Variational Autoencoder for clustering.

        VAE learns a probabilistic latent space with KL divergence regularization,
        which can produce more structured representations than standard autoencoders.

        Steps:
        1. Train VAE with reconstruction loss + KL divergence
        2. Extract latent representations (mean of posterior)
        3. Apply K-Means clustering on latent space
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.warning("PyTorch not installed, falling back to K-Means")
            return self._fit_kmeans()

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = self._X_scaled.shape[1]
        latent_dim = min(self.ae_latent_dim, input_dim - 1)
        hidden_dim = self.ae_hidden_dims[0] if self.ae_hidden_dims else 32

        # VAE Model Definition
        class VAE(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                # Encoder
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
                # Decoder
                self.fc3 = nn.Linear(latent_dim, hidden_dim)
                self.bn3 = nn.BatchNorm1d(hidden_dim)
                self.fc4 = nn.Linear(hidden_dim, input_dim)

            def encode(self, x):
                h = F.relu(self.bn1(self.fc1(x)))
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                h = F.relu(self.bn3(self.fc3(z)))
                return self.fc4(h)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                recon = self.decode(z)
                return recon, mu, logvar

        vae = VAE(input_dim, hidden_dim, latent_dim)

        # Training setup
        X_tensor = torch.FloatTensor(self._X_scaled)
        dataset = TensorDataset(X_tensor, X_tensor)
        batch_size = min(self.ae_batch_size, len(X_tensor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(vae.parameters(), lr=self.ae_learning_rate)

        def vae_loss(recon_x, x, mu, logvar, beta=1.0):
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + beta * kl_loss

        # Training loop
        vae.train()
        for epoch in range(self.ae_epochs):
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                recon, mu, logvar = vae(batch_x)
                loss = vae_loss(recon, batch_x, mu, logvar)
                loss.backward()
                optimizer.step()

        # Extract latent representations (use mean)
        vae.eval()
        with torch.no_grad():
            mu, _ = vae.encode(X_tensor)
            latent_repr = mu.numpy()

        # Cluster in latent space
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(latent_repr)

        self.model = {'vae': vae, 'kmeans': kmeans, 'latent_dim': latent_dim}
        logger.debug(f"VAE: compressed {input_dim}D -> {latent_dim}D latent space")

        return labels

    def _fit_sparse_autoencoder(self) -> np.ndarray:
        """
        Sparse Autoencoder for clustering.

        Adds sparsity regularization (L1 penalty) on the latent representation,
        encouraging the model to learn more interpretable features.

        Steps:
        1. Train autoencoder with reconstruction loss + sparsity penalty
        2. Extract sparse latent representations
        3. Apply K-Means clustering on latent space
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.warning("PyTorch not installed, falling back to K-Means")
            return self._fit_kmeans()

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = self._X_scaled.shape[1]
        latent_dim = min(self.ae_latent_dim, input_dim - 1)
        hidden_dim = self.ae_hidden_dims[0] if self.ae_hidden_dims else 32

        class SparseAutoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.Sigmoid(),  # Bounded activation for sparsity
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def forward(self, x):
                z = self.encoder(x)
                recon = self.decoder(z)
                return recon, z

            def encode(self, x):
                return self.encoder(x)

        model = SparseAutoencoder(input_dim, hidden_dim, latent_dim)

        # Training setup
        X_tensor = torch.FloatTensor(self._X_scaled)
        dataset = TensorDataset(X_tensor, X_tensor)
        batch_size = min(self.ae_batch_size, len(X_tensor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.ae_learning_rate)
        mse_loss = nn.MSELoss()

        # Sparsity parameters
        sparsity_target = 0.05  # Target average activation
        sparsity_weight = 0.1

        # Training loop
        model.train()
        for epoch in range(self.ae_epochs):
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                recon, z = model(batch_x)

                # Reconstruction loss
                recon_loss = mse_loss(recon, batch_x)

                # Sparsity loss (KL divergence from target sparsity)
                avg_activation = z.mean(dim=0)
                sparsity_loss = torch.sum(
                    sparsity_target * torch.log(sparsity_target / (avg_activation + 1e-8)) +
                    (1 - sparsity_target) * torch.log((1 - sparsity_target) / (1 - avg_activation + 1e-8))
                )

                loss = recon_loss + sparsity_weight * sparsity_loss
                loss.backward()
                optimizer.step()

        # Extract latent representations
        model.eval()
        with torch.no_grad():
            latent_repr = model.encode(X_tensor).numpy()

        # Cluster in latent space
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(latent_repr)

        self.model = {'sparse_ae': model, 'kmeans': kmeans, 'latent_dim': latent_dim}
        logger.debug(f"SparseAE: compressed {input_dim}D -> {latent_dim}D latent space")

        return labels

    def _fit_deep_infomax(self) -> np.ndarray:
        """
        Deep InfoMax (DIM) for clustering.

        Maximizes mutual information between input and latent representations,
        learning features that preserve the most information about the input.

        Steps:
        1. Train encoder + discriminator to maximize MI
        2. Extract latent representations
        3. Apply K-Means clustering on latent space
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.warning("PyTorch not installed, falling back to K-Means")
            return self._fit_kmeans()

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = self._X_scaled.shape[1]
        latent_dim = min(self.ae_latent_dim, input_dim - 1)
        hidden_dim = self.ae_hidden_dims[0] if self.ae_hidden_dims else 32

        class Encoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, latent_dim),
                )

            def forward(self, x):
                return self.net(x)

        class Discriminator(nn.Module):
            """Discriminator for global MI estimation."""
            def __init__(self, input_dim, latent_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim + latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )

            def forward(self, x, z):
                xz = torch.cat([x, z], dim=1)
                return self.net(xz)

        encoder = Encoder(input_dim, hidden_dim, latent_dim)
        discriminator = Discriminator(input_dim, latent_dim, hidden_dim)

        # Training setup
        X_tensor = torch.FloatTensor(self._X_scaled)
        dataset = TensorDataset(X_tensor)
        batch_size = min(self.ae_batch_size, len(X_tensor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(discriminator.parameters()),
            lr=self.ae_learning_rate
        )

        # Training loop
        encoder.train()
        discriminator.train()
        for epoch in range(self.ae_epochs):
            for (batch_x,) in dataloader:
                optimizer.zero_grad()

                # Positive samples (matched pairs)
                z = encoder(batch_x)
                pos_scores = discriminator(batch_x, z)

                # Negative samples (shuffled pairs)
                perm = torch.randperm(batch_x.size(0))
                neg_scores = discriminator(batch_x, z[perm])

                # InfoNCE loss
                loss = -torch.mean(
                    F.logsigmoid(pos_scores) + F.logsigmoid(-neg_scores)
                )

                loss.backward()
                optimizer.step()

        # Extract latent representations
        encoder.eval()
        with torch.no_grad():
            latent_repr = encoder(X_tensor).numpy()

        # Cluster in latent space
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(latent_repr)

        self.model = {'encoder': encoder, 'discriminator': discriminator, 'kmeans': kmeans}
        logger.debug(f"DeepInfoMax: learned {latent_dim}D representation via MI maximization")

        return labels

    def _fit_bigan(self) -> np.ndarray:
        """
        Bidirectional GAN (BiGAN) for clustering.

        Learns an encoder and generator simultaneously through adversarial training,
        producing representations that capture data distribution structure.

        Steps:
        1. Train encoder, generator, and discriminator adversarially
        2. Extract latent representations via encoder
        3. Apply K-Means clustering on latent space
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.warning("PyTorch not installed, falling back to K-Means")
            return self._fit_kmeans()

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = self._X_scaled.shape[1]
        latent_dim = min(self.ae_latent_dim, input_dim - 1)
        hidden_dim = self.ae_hidden_dims[0] if self.ae_hidden_dims else 32

        class Encoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, latent_dim),
                )

            def forward(self, x):
                return self.net(x)

        class Generator(nn.Module):
            def __init__(self, latent_dim, hidden_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, output_dim),
                )

            def forward(self, z):
                return self.net(z)

        class Discriminator(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim + latent_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x, z):
                xz = torch.cat([x, z], dim=1)
                return self.net(xz)

        encoder = Encoder(input_dim, hidden_dim, latent_dim)
        generator = Generator(latent_dim, hidden_dim, input_dim)
        discriminator = Discriminator(input_dim, latent_dim, hidden_dim)

        # Training setup
        X_tensor = torch.FloatTensor(self._X_scaled)
        dataset = TensorDataset(X_tensor)
        batch_size = min(self.ae_batch_size, len(X_tensor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt_enc_gen = torch.optim.Adam(
            list(encoder.parameters()) + list(generator.parameters()),
            lr=self.ae_learning_rate, betas=(0.5, 0.999)
        )
        opt_disc = torch.optim.Adam(
            discriminator.parameters(),
            lr=self.ae_learning_rate, betas=(0.5, 0.999)
        )

        criterion = nn.BCELoss()

        # Training loop
        for epoch in range(self.ae_epochs):
            for (batch_x,) in dataloader:
                batch_size_actual = batch_x.size(0)

                # Labels
                real_label = torch.ones(batch_size_actual, 1)
                fake_label = torch.zeros(batch_size_actual, 1)

                # ---- Train Discriminator ----
                opt_disc.zero_grad()

                # Real: (x, E(x))
                z_real = encoder(batch_x)
                d_real = discriminator(batch_x, z_real.detach())
                loss_d_real = criterion(d_real, real_label)

                # Fake: (G(z), z)
                z_fake = torch.randn(batch_size_actual, latent_dim)
                x_fake = generator(z_fake)
                d_fake = discriminator(x_fake.detach(), z_fake)
                loss_d_fake = criterion(d_fake, fake_label)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                opt_disc.step()

                # ---- Train Encoder/Generator ----
                opt_enc_gen.zero_grad()

                z_real = encoder(batch_x)
                d_real = discriminator(batch_x, z_real)
                loss_e = criterion(d_real, fake_label)  # Encoder tries to fool D

                z_fake = torch.randn(batch_size_actual, latent_dim)
                x_fake = generator(z_fake)
                d_fake = discriminator(x_fake, z_fake)
                loss_g = criterion(d_fake, real_label)  # Generator tries to fool D

                loss_eg = loss_e + loss_g
                loss_eg.backward()
                opt_enc_gen.step()

        # Extract latent representations
        encoder.eval()
        with torch.no_grad():
            latent_repr = encoder(X_tensor).numpy()

        # Cluster in latent space
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(latent_repr)

        self.model = {'encoder': encoder, 'generator': generator, 'discriminator': discriminator, 'kmeans': kmeans}
        logger.debug(f"BiGAN: learned {latent_dim}D representation via adversarial training")

        return labels

    def _fit_iic(self) -> np.ndarray:
        """
        Invariant Information Clustering (IIC).

        Directly optimizes cluster assignments by maximizing mutual information
        between cluster assignments of original and augmented samples.

        Steps:
        1. Train network to output cluster probabilities
        2. Maximize MI between assignments of paired samples
        3. Return cluster assignments (argmax of probabilities)
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.warning("PyTorch not installed, falling back to K-Means")
            return self._fit_kmeans()

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = self._X_scaled.shape[1]
        hidden_dim = self.ae_hidden_dims[0] if self.ae_hidden_dims else 32
        n_clusters = self.n_clusters

        class IICNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, n_clusters):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, n_clusters),
                )

            def forward(self, x):
                return F.softmax(self.net(x), dim=1)

        def iic_loss(p1, p2, eps=1e-8):
            """Compute IIC loss (negative MI between cluster assignments)."""
            # Joint probability P(c1, c2)
            p_joint = torch.mm(p1.t(), p2) / p1.size(0)
            # Marginals
            p1_marginal = p1.mean(dim=0, keepdim=True)
            p2_marginal = p2.mean(dim=0, keepdim=True)
            # MI = sum P(c1,c2) * log(P(c1,c2) / (P(c1)*P(c2)))
            mi = (p_joint * torch.log((p_joint + eps) / (p1_marginal.t() @ p2_marginal + eps))).sum()
            return -mi  # Negative because we minimize

        def augment_data(x, noise_std=0.1):
            """Simple augmentation: add Gaussian noise."""
            return x + torch.randn_like(x) * noise_std

        model = IICNetwork(input_dim, hidden_dim, n_clusters)

        # Training setup
        X_tensor = torch.FloatTensor(self._X_scaled)
        dataset = TensorDataset(X_tensor)
        batch_size = min(self.ae_batch_size, len(X_tensor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.ae_learning_rate)

        # Training loop
        model.train()
        for epoch in range(self.ae_epochs):
            for (batch_x,) in dataloader:
                optimizer.zero_grad()

                # Original and augmented views
                p1 = model(batch_x)
                p2 = model(augment_data(batch_x))

                loss = iic_loss(p1, p2)
                loss.backward()
                optimizer.step()

        # Get cluster assignments
        model.eval()
        with torch.no_grad():
            probs = model(X_tensor).numpy()
        labels = probs.argmax(axis=1)

        self.model = {'iic_net': model}
        logger.debug(f"IIC: direct clustering into {n_clusters} clusters via MI maximization")

        return labels

    def _fit_dac(self) -> np.ndarray:
        """
        Deep Adaptive Clustering (DAC).

        Learns clustering by training on pairwise similarity predictions,
        using cosine similarity between learned representations.

        Steps:
        1. Initialize with K-Means on raw features
        2. Train network to predict pairwise similarities
        3. Iteratively refine clusters and network
        4. Return final cluster assignments
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.warning("PyTorch not installed, falling back to K-Means")
            return self._fit_kmeans()

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = self._X_scaled.shape[1]
        latent_dim = min(self.ae_latent_dim, input_dim - 1)
        hidden_dim = self.ae_hidden_dims[0] if self.ae_hidden_dims else 32
        n_clusters = self.n_clusters

        class DACNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, latent_dim),
                )

            def forward(self, x):
                z = self.encoder(x)
                # L2 normalize for cosine similarity
                z = F.normalize(z, p=2, dim=1)
                return z

        model = DACNetwork(input_dim, hidden_dim, latent_dim)

        # Initialize with K-Means
        kmeans_init = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        initial_labels = kmeans_init.fit_predict(self._X_scaled)

        # Training setup
        X_tensor = torch.FloatTensor(self._X_scaled)
        dataset = TensorDataset(X_tensor)
        batch_size = min(self.ae_batch_size, len(X_tensor))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.ae_learning_rate)

        # Training parameters
        lambda_high = 0.95  # High confidence threshold for positive pairs
        lambda_low = 0.05   # Low confidence threshold for negative pairs

        labels_tensor = torch.LongTensor(initial_labels)

        # Training loop
        model.train()
        for epoch in range(self.ae_epochs):
            # Get current representations
            model.eval()
            with torch.no_grad():
                z_all = model(X_tensor)
                # Compute pairwise cosine similarities
                sim_matrix = torch.mm(z_all, z_all.t())

            model.train()

            # Sample pairs for training
            indices = torch.randperm(len(X_tensor))[:min(batch_size * 4, len(X_tensor))]

            for i in range(0, len(indices), 2):
                if i + 1 >= len(indices):
                    break

                idx1, idx2 = indices[i].item(), indices[i + 1].item()
                x1, x2 = X_tensor[idx1:idx1+1], X_tensor[idx2:idx2+1]

                optimizer.zero_grad()

                z1 = model(x1)
                z2 = model(x2)
                sim = F.cosine_similarity(z1, z2)

                # Determine label based on initial clustering + confidence
                same_cluster = (labels_tensor[idx1] == labels_tensor[idx2]).float()

                # Adaptive threshold based on similarity confidence
                current_sim = sim_matrix[idx1, idx2].item()
                if current_sim > lambda_high:
                    target = 1.0
                elif current_sim < lambda_low:
                    target = 0.0
                else:
                    target = same_cluster.item()

                target = torch.tensor([[target]])
                loss = F.mse_loss(sim.unsqueeze(0), target)
                loss.backward()
                optimizer.step()

            # Update cluster assignments periodically
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    z_all = model(X_tensor).numpy()
                kmeans_temp = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=3)
                labels_tensor = torch.LongTensor(kmeans_temp.fit_predict(z_all))
                model.train()

        # Final clustering
        model.eval()
        with torch.no_grad():
            latent_repr = model(X_tensor).numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(latent_repr)

        self.model = {'dac_net': model, 'kmeans': kmeans}
        logger.debug(f"DAC: learned {latent_dim}D representation via pairwise similarity learning")

        return labels

    def _compute_metrics(self, labels: np.ndarray) -> ClusterResult:
        """Calcula métricas de calidad del clustering."""
        # Filtrar ruido para métricas (label == -1)
        non_noise_mask = labels >= 0
        n_noise = (~non_noise_mask).sum()

        # Calcular métricas solo si hay suficientes clusters
        unique_labels = set(labels[non_noise_mask])
        n_clusters = len(unique_labels)

        if n_clusters >= 2 and non_noise_mask.sum() > n_clusters:
            X_valid = self._X_scaled[non_noise_mask]
            labels_valid = labels[non_noise_mask]

            silhouette = silhouette_score(X_valid, labels_valid)
            calinski = calinski_harabasz_score(X_valid, labels_valid)
            davies = davies_bouldin_score(X_valid, labels_valid)
        else:
            silhouette = -1
            calinski = 0
            davies = float('inf')

        # Tamaño de cada cluster
        cluster_sizes = {}
        for label in sorted(set(labels)):
            cluster_sizes[label] = (labels == label).sum()

        return ClusterResult(
            method=self.method,
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            silhouette=silhouette,
            calinski_harabasz=calinski,
            davies_bouldin=davies,
            cluster_sizes=cluster_sizes,
            model=self.model,
        )

    @staticmethod
    def find_optimal_k(
        algo_features: pd.DataFrame,
        method: ClusterMethod = ClusterMethod.KMEANS,
        k_range: range = range(2, 11),
        features: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Encuentra número óptimo de clusters probando varios valores de k.

        Returns:
            DataFrame con métricas por valor de k
        """
        results = []

        for k in k_range:
            clusterer = AlgoClusterer(
                method=method,
                n_clusters=k,
                features=features,
            )
            result = clusterer.fit(algo_features)

            results.append({
                'k': k,
                'silhouette': result.silhouette,
                'calinski_harabasz': result.calinski_harabasz,
                'davies_bouldin': result.davies_bouldin,
            })

        return pd.DataFrame(results).set_index('k')

    @staticmethod
    def compare_methods(
        algo_features: pd.DataFrame,
        n_clusters: int = 5,
        features: Optional[list[str]] = None,
        scaler_type: ScalerType = ScalerType.ROBUST,
    ) -> dict[ClusterMethod, ClusterResult]:
        """
        Compara todos los métodos de clustering.

        Returns:
            Dict {method: ClusterResult}
        """
        results = {}

        for method in ClusterMethod:
            try:
                clusterer = AlgoClusterer(
                    method=method,
                    n_clusters=n_clusters,
                    features=features,
                    scaler_type=scaler_type,
                )
                results[method] = clusterer.fit(algo_features)
            except Exception as e:
                logger.error(f"Error with {method.value}: {e}")

        return results

    @staticmethod
    def two_layer_clustering(
        algo_features: pd.DataFrame,
        life_features: list[str],
        behavior_features: list[str],
        life_method: ClusterMethod = ClusterMethod.HDBSCAN,
        behavior_method: ClusterMethod = ClusterMethod.GMM,
        n_life_clusters: int = 4,
        n_behavior_clusters: int = 3,
        min_cluster_size_for_subclustering: int = 50,
        scaler_type: ScalerType = ScalerType.ROBUST,
    ) -> TwoLayerClusterResult:
        """
        Clustering de dos capas: primero por perfil de vida, luego por comportamiento.

        Capa 1 (Life Profile):
            Agrupa algoritmos por su patrón de actividad temporal:
            - Cuándo empiezan
            - Cuánto duran
            - Qué porcentaje del estudio están activos

        Capa 2 (Financial Behavior):
            Dentro de cada grupo de vida, agrupa por comportamiento financiero:
            - Rendimiento
            - Riesgo
            - Estabilidad
            - Evolución temporal

        Args:
            algo_features: DataFrame con features
            life_features: Features para capa 1 (actividad)
            behavior_features: Features para capa 2 (rendimiento)
            life_method: Método para capa 1 (HDBSCAN recomendado)
            behavior_method: Método para capa 2 (GMM recomendado)
            n_life_clusters: Número de clusters de vida
            n_behavior_clusters: Número de clusters de comportamiento por grupo
            min_cluster_size_for_subclustering: Mínimo tamaño para sub-clustering
            scaler_type: Tipo de escalado

        Returns:
            TwoLayerClusterResult con resultados de ambas capas
        """
        logger.info("="*60)
        logger.info("CLUSTERING DE DOS CAPAS")
        logger.info("="*60)

        # ==========================================
        # CAPA 1: LIFE PROFILE
        # ==========================================
        logger.info("\n--- Capa 1: Life Profile ---")
        logger.info(f"Features: {life_features}")
        logger.info(f"Método: {life_method.value}")

        life_clusterer = AlgoClusterer(
            method=life_method,
            n_clusters=n_life_clusters,
            features=life_features,
            scaler_type=scaler_type,
            min_cluster_size=max(15, len(algo_features) // 100),
        )
        life_result = life_clusterer.fit(algo_features)

        # Nombrar clusters de vida
        life_names = name_life_clusters(algo_features, life_result.labels, life_features)

        logger.info(f"Life clusters encontrados: {life_result.n_clusters}")
        for cluster_id, name in sorted(life_names.items()):
            count = life_result.cluster_sizes.get(cluster_id, 0)
            logger.info(f"  {cluster_id}: {name} ({count} algos)")

        # ==========================================
        # CAPA 2: FINANCIAL BEHAVIOR (por cada life profile)
        # ==========================================
        logger.info("\n--- Capa 2: Financial Behavior ---")
        logger.info(f"Features: {behavior_features}")
        logger.info(f"Método: {behavior_method.value}")

        behavior_results = {}
        behavior_names = {}

        # Añadir life cluster a los features
        algo_features_with_life = algo_features.copy()
        algo_features_with_life['life_cluster'] = life_result.labels

        for life_cluster_id in sorted(set(life_result.labels)):
            if life_cluster_id == -1:  # Skip noise
                continue

            # Filtrar algoritmos de este life cluster
            mask = algo_features_with_life['life_cluster'] == life_cluster_id
            cluster_algos = algo_features_with_life[mask]

            life_name = life_names.get(life_cluster_id, f"life_{life_cluster_id}")
            logger.info(f"\n  Sub-clustering '{life_name}' ({len(cluster_algos)} algos)")

            # Solo sub-clusterizar si hay suficientes algoritmos
            if len(cluster_algos) < min_cluster_size_for_subclustering:
                logger.info(f"    Muy pocos algoritmos, asignando cluster único")
                behavior_results[life_cluster_id] = None
                behavior_names[life_cluster_id] = {0: "único"}
                continue

            # Ajustar número de clusters según tamaño del grupo
            adjusted_n_clusters = min(
                n_behavior_clusters,
                max(2, len(cluster_algos) // 30)
            )

            try:
                behavior_clusterer = AlgoClusterer(
                    method=behavior_method,
                    n_clusters=adjusted_n_clusters,
                    features=behavior_features,
                    scaler_type=scaler_type,
                )
                behavior_result = behavior_clusterer.fit(cluster_algos)
                behavior_results[life_cluster_id] = behavior_result

                # Nombrar clusters de comportamiento
                beh_names = name_behavior_clusters(
                    cluster_algos, behavior_result.labels, behavior_features
                )
                behavior_names[life_cluster_id] = beh_names

                logger.info(f"    Clusters: {behavior_result.n_clusters}, "
                           f"silhouette: {behavior_result.silhouette:.3f}")

            except Exception as e:
                logger.warning(f"    Error en sub-clustering: {e}")
                behavior_results[life_cluster_id] = None
                behavior_names[life_cluster_id] = {0: "error"}

        # ==========================================
        # COMBINAR LABELS
        # ==========================================
        combined_labels = _combine_cluster_labels(
            algo_features,
            life_result.labels,
            behavior_results,
            life_names,
            behavior_names,
        )

        # Calcular métricas globales
        n_total_clusters = combined_labels.nunique()

        # Silhouette global (aproximado)
        overall_silhouette = life_result.silhouette  # Usar de capa 1 como proxy

        logger.info("\n" + "="*60)
        logger.info(f"RESUMEN: {n_total_clusters} clusters combinados")
        logger.info("="*60)

        return TwoLayerClusterResult(
            life_profile_result=life_result,
            life_profile_names=life_names,
            behavior_results=behavior_results,
            behavior_names=behavior_names,
            combined_labels=combined_labels,
            overall_silhouette=overall_silhouette,
            n_total_clusters=n_total_clusters,
        )


def _combine_cluster_labels(
    algo_features: pd.DataFrame,
    life_labels: np.ndarray,
    behavior_results: dict[int, Optional[ClusterResult]],
    life_names: dict[int, str],
    behavior_names: dict[int, dict[int, str]],
) -> pd.Series:
    """Combina labels de ambas capas en un label único."""
    combined = []

    for i, (algo_id, life_label) in enumerate(zip(algo_features.index, life_labels)):
        if life_label == -1:
            combined.append("noise")
            continue

        life_name = life_names.get(life_label, f"L{life_label}")

        # Obtener behavior label
        behavior_result = behavior_results.get(life_label)
        if behavior_result is None:
            behavior_name = behavior_names.get(life_label, {}).get(0, "único")
        else:
            # Encontrar índice relativo del algoritmo en su life cluster
            life_mask = life_labels == life_label
            life_indices = np.where(life_mask)[0]
            relative_idx = np.where(life_indices == i)[0]

            if len(relative_idx) > 0:
                behavior_label = behavior_result.labels[relative_idx[0]]
                behavior_name = behavior_names.get(life_label, {}).get(
                    behavior_label, f"B{behavior_label}"
                )
            else:
                behavior_name = "unknown"

        combined.append(f"{life_name}__{behavior_name}")

    return pd.Series(combined, index=algo_features.index, name='cluster')


def name_life_clusters(
    algo_features: pd.DataFrame,
    labels: np.ndarray,
    life_features: list[str],
) -> dict[int, str]:
    """
    Asigna nombres descriptivos a clusters de perfil de vida.

    Nombres basados en:
    - Temporalidad: early, mid, late (cuándo empiezan)
    - Duración: short, medium, long
    - Persistencia: persistent, intermittent
    """
    df = algo_features.copy()
    df['cluster'] = labels

    # Calcular medias por cluster
    cluster_means = df.groupby('cluster')[life_features].mean()

    names = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            names[cluster_id] = "noise"
            continue

        row = cluster_means.loc[cluster_id]
        parts = []

        # Temporalidad (cuándo empiezan)
        if 'start_idx' in row:
            if row['start_idx'] < 0.2:
                parts.append("early")
            elif row['start_idx'] > 0.6:
                parts.append("late")
            else:
                parts.append("mid")

        # Duración
        if 'duration_ratio' in row:
            if row['duration_ratio'] > 0.7:
                parts.append("long")
            elif row['duration_ratio'] < 0.3:
                parts.append("short")
            else:
                parts.append("medium")

        # Persistencia
        if 'active_ratio' in row:
            if row['active_ratio'] > 0.9:
                parts.append("persistent")
            elif row['active_ratio'] < 0.7:
                parts.append("intermittent")

        names[cluster_id] = "_".join(parts) if parts else f"life_{cluster_id}"

    # Resolver duplicados
    return _resolve_duplicate_names(names)


def name_behavior_clusters(
    algo_features: pd.DataFrame,
    labels: np.ndarray,
    behavior_features: list[str],
) -> dict[int, str]:
    """
    Asigna nombres descriptivos a clusters de comportamiento financiero.

    Nombres basados en:
    - Performance: high_perf, low_perf
    - Riesgo: low_risk, high_risk
    - Estabilidad: stable, unstable
    - Evolución: improving, degrading
    """
    df = algo_features.copy()
    df['cluster'] = labels

    # Calcular medias por cluster
    available_features = [f for f in behavior_features if f in df.columns]
    if not available_features:
        return {i: f"beh_{i}" for i in sorted(set(labels)) if i >= 0}

    cluster_means = df.groupby('cluster')[available_features].mean()

    names = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            names[cluster_id] = "noise"
            continue

        row = cluster_means.loc[cluster_id]
        parts = []

        # Performance
        if 'sharpe' in row:
            median_sharpe = cluster_means['sharpe'].median()
            if row['sharpe'] > median_sharpe:
                parts.append("high_perf")
            else:
                parts.append("low_perf")

        # Riesgo
        if 'max_dd' in row:
            median_dd = cluster_means['max_dd'].median()
            if row['max_dd'] > median_dd:  # max_dd es negativo
                parts.append("low_risk")
            else:
                parts.append("high_risk")

        # Estabilidad
        if 'sharpe_stability' in row:
            median_stab = cluster_means['sharpe_stability'].median()
            if row['sharpe_stability'] > median_stab:
                parts.append("stable")
            else:
                parts.append("volatile")

        # Evolución temporal
        if 'return_decay' in row:
            if row['return_decay'] > 0.05:
                parts.append("improving")
            elif row['return_decay'] < -0.05:
                parts.append("degrading")

        names[cluster_id] = "_".join(parts) if parts else f"beh_{cluster_id}"

    # Resolver duplicados
    return _resolve_duplicate_names(names)


def name_clusters(
    algo_features: pd.DataFrame,
    labels: np.ndarray,
    clustering_features: list[str],
) -> dict[int, str]:
    """
    Asigna nombres descriptivos a cada cluster basado en sus características.
    (Versión legacy para compatibilidad)

    Args:
        algo_features: DataFrame con features
        labels: Array de labels de cluster
        clustering_features: Features usadas para clustering

    Returns:
        Dict {cluster_id: nombre}
    """
    algo_features = algo_features.copy()
    algo_features['cluster'] = labels

    # Calcular medias por cluster
    available_features = [f for f in clustering_features if f in algo_features.columns]
    cluster_means = algo_features.groupby('cluster')[available_features].mean()

    # Función para generar nombre
    def generate_name(row, cluster_means):
        names = []

        # Performance
        if 'sharpe' in row:
            if row['sharpe'] > cluster_means['sharpe'].median():
                names.append('high_perf')
            else:
                names.append('low_perf')

        # Volatility
        if 'ann_vol' in row:
            if row['ann_vol'] > cluster_means['ann_vol'].median():
                names.append('high_vol')
            else:
                names.append('low_vol')

        # Trend
        if 'trend_score' in row:
            if row['trend_score'] > 0.02:
                names.append('trend')
            elif row['trend_score'] < -0.02:
                names.append('reversal')

        # Correlation
        if 'corr_benchmark' in row:
            if row['corr_benchmark'] > cluster_means['corr_benchmark'].quantile(0.75):
                names.append('high_corr')
            elif row['corr_benchmark'] < cluster_means['corr_benchmark'].quantile(0.25):
                names.append('low_corr')

        # Decay/Stability
        if 'return_decay' in row:
            if row['return_decay'] > 0.05:
                names.append('improving')
            elif row['return_decay'] < -0.05:
                names.append('degrading')

        # Duration
        if 'duration_ratio' in row:
            if row['duration_ratio'] > 0.7:
                names.append('long_lived')
            elif row['duration_ratio'] < 0.3:
                names.append('short_lived')

        return '_'.join(names) if names else f'cluster_{int(row.name)}'

    # Generar nombres iniciales
    initial_names = {}
    for cluster_id, row in cluster_means.iterrows():
        if cluster_id == -1:  # Ruido
            initial_names[cluster_id] = 'noise'
        else:
            initial_names[cluster_id] = generate_name(row, cluster_means)

    return _resolve_duplicate_names(initial_names)


def _resolve_duplicate_names(names: dict[int, str]) -> dict[int, str]:
    """Resuelve nombres duplicados añadiendo sufijos numéricos."""
    name_counts = {}
    resolved = {}

    for cluster_id in sorted(names.keys()):
        base_name = names[cluster_id]
        if base_name in name_counts:
            name_counts[base_name] += 1
            resolved[cluster_id] = f"{base_name}_{name_counts[base_name]}"
        else:
            name_counts[base_name] = 1
            resolved[cluster_id] = base_name

    return resolved


# =============================================================================
# TEMPORAL ALGO CLUSTERER - Weekly clustering with 3 time horizons
# =============================================================================

@dataclass
class TemporalClusterResult:
    """Result of temporal clustering for a single week."""
    week_end: pd.Timestamp
    n_algos: int
    n_active: int  # Algos with data in this period

    # Cluster assignments per algo (index = algo_id)
    cluster_cumulative: pd.Series  # From start to week_end
    cluster_weekly: pd.Series      # Current week only
    cluster_monthly: pd.Series     # Current month

    # Features used
    features_cumulative: pd.DataFrame
    features_weekly: pd.DataFrame
    features_monthly: pd.DataFrame

    # Clustering quality metrics
    metrics_cumulative: dict
    metrics_weekly: dict
    metrics_monthly: dict


@dataclass
class TemporalClusteringOutput:
    """Full output from temporal clustering analysis."""
    # All weekly results
    weekly_results: list[TemporalClusterResult]

    # Method comparison (per time horizon)
    method_comparison: dict[str, pd.DataFrame]  # {horizon: comparison_df}

    # Best methods by metric
    best_methods: dict[str, ClusterMethod]  # {horizon: method}

    # Full cluster history DataFrame
    cluster_history: pd.DataFrame  # Columns: week_end, algo_id, cluster_cumulative, cluster_weekly, cluster_monthly

    # Parameters used
    params: dict


class TemporalAlgoClusterer:
    """
    Temporal clustering of algorithms with three time horizons.

    For each week, clusters algorithms based on:
    1. Cumulative performance (from start_date to current week)
    2. Weekly performance (current week only)
    3. Monthly performance (current month)

    Handles missing algorithms appropriately by:
    - Excluding from clustering if no data in period
    - Marking as 'inactive' in results

    Usage:
        clusterer = TemporalAlgoClusterer(
            returns_matrix=algo_returns,  # DataFrame [dates x algos]
            start_date='2020-01-01',
            n_clusters=5,
        )
        output = clusterer.run()

        # Compare clustering methods
        comparison = clusterer.compare_all_methods()

        # Save results
        clusterer.save_results('data/processed/temporal_clusters/')
    """

    # Features to compute for clustering
    CLUSTERING_FEATURES = [
        'return',
        'volatility',
        'sharpe',
        'max_drawdown',
        'calmar_ratio',
        'profit_factor',
    ]

    def __init__(
        self,
        returns_matrix: pd.DataFrame,
        start_date: str = '2020-01-01',
        n_clusters: int = 5,
        method: ClusterMethod = ClusterMethod.KMEANS,
        scaler_type: ScalerType = ScalerType.ROBUST,
        min_data_points: int = 5,  # Min data points to include in clustering
        random_state: int = 42,
    ):
        """
        Initialize temporal clusterer.

        Args:
            returns_matrix: DataFrame with daily returns [dates x algos]
            start_date: Start date for cumulative calculations
            n_clusters: Number of clusters
            method: Clustering method to use
            scaler_type: Scaler for feature normalization
            min_data_points: Minimum data points required for an algo to be clustered
            random_state: Random seed for reproducibility
        """
        self.returns_matrix = returns_matrix.copy()
        self.start_date = pd.Timestamp(start_date)
        self.n_clusters = n_clusters
        self.method = method
        self.scaler_type = scaler_type
        self.min_data_points = min_data_points
        self.random_state = random_state

        # Ensure datetime index
        if not isinstance(self.returns_matrix.index, pd.DatetimeIndex):
            self.returns_matrix.index = pd.to_datetime(self.returns_matrix.index)

        # Filter from start date
        self.returns_matrix = self.returns_matrix[
            self.returns_matrix.index >= self.start_date
        ]

        # Get week ends (Friday) for analysis
        self.week_ends = self._get_week_ends()

        logger.info(
            f"TemporalAlgoClusterer initialized: {len(self.returns_matrix.columns)} algos, "
            f"{len(self.week_ends)} weeks from {self.start_date.date()}"
        )

    def _get_week_ends(self) -> list[pd.Timestamp]:
        """Get all Friday dates in the data."""
        # Resample to weekly (Friday) and get last date of each week
        weekly = self.returns_matrix.resample('W-FRI').last()
        return list(weekly.index)

    def run(
        self,
        methods: Optional[list[ClusterMethod]] = None,
        save_path: Optional[str] = None,
    ) -> TemporalClusteringOutput:
        """
        Run temporal clustering for all weeks.

        Args:
            methods: List of methods to test (default: all)
            save_path: Path to save results (optional)

        Returns:
            TemporalClusteringOutput with all results
        """
        if methods is None:
            methods = [self.method]

        logger.info(f"Running temporal clustering with methods: {[m.value for m in methods]}")

        # Run clustering for all weeks with primary method
        weekly_results = []
        cluster_history_rows = []

        for week_idx, week_end in enumerate(self.week_ends):
            if week_idx % 10 == 0:
                logger.info(f"Processing week {week_idx + 1}/{len(self.week_ends)}: {week_end.date()}")

            result = self._cluster_single_week(week_end)
            weekly_results.append(result)

            # Build history rows
            for algo_id in self.returns_matrix.columns:
                cluster_history_rows.append({
                    'week_end': week_end,
                    'algo_id': algo_id,
                    'cluster_cumulative': result.cluster_cumulative.get(algo_id, 'inactive'),
                    'cluster_weekly': result.cluster_weekly.get(algo_id, 'inactive'),
                    'cluster_monthly': result.cluster_monthly.get(algo_id, 'inactive'),
                })

        cluster_history = pd.DataFrame(cluster_history_rows)

        # Compare methods if multiple specified
        method_comparison = {}
        best_methods = {}

        if len(methods) > 1:
            method_comparison = self.compare_all_methods(methods)
            best_methods = self._select_best_methods(method_comparison)
        else:
            best_methods = {
                'cumulative': self.method,
                'weekly': self.method,
                'monthly': self.method,
            }

        output = TemporalClusteringOutput(
            weekly_results=weekly_results,
            method_comparison=method_comparison,
            best_methods=best_methods,
            cluster_history=cluster_history,
            params={
                'start_date': str(self.start_date.date()),
                'n_clusters': self.n_clusters,
                'method': self.method.value,
                'n_weeks': len(self.week_ends),
                'n_algos': len(self.returns_matrix.columns),
            },
        )

        if save_path:
            self.save_results(output, save_path)

        return output

    def _cluster_single_week(self, week_end: pd.Timestamp) -> TemporalClusterResult:
        """Cluster algorithms for a single week."""
        # Define time periods
        week_start = week_end - pd.Timedelta(days=6)
        month_start = week_end.replace(day=1)

        # Compute features for each horizon
        features_cumulative = self._compute_period_features(
            self.start_date, week_end, 'cumulative'
        )
        features_weekly = self._compute_period_features(
            week_start, week_end, 'weekly'
        )
        features_monthly = self._compute_period_features(
            month_start, week_end, 'monthly'
        )

        # Cluster each horizon
        cluster_cumulative, metrics_cumulative = self._cluster_features(
            features_cumulative, 'cumulative'
        )
        cluster_weekly, metrics_weekly = self._cluster_features(
            features_weekly, 'weekly'
        )
        cluster_monthly, metrics_monthly = self._cluster_features(
            features_monthly, 'monthly'
        )

        # Count active algos
        n_active = len(features_cumulative.dropna(how='all'))

        return TemporalClusterResult(
            week_end=week_end,
            n_algos=len(self.returns_matrix.columns),
            n_active=n_active,
            cluster_cumulative=cluster_cumulative,
            cluster_weekly=cluster_weekly,
            cluster_monthly=cluster_monthly,
            features_cumulative=features_cumulative,
            features_weekly=features_weekly,
            features_monthly=features_monthly,
            metrics_cumulative=metrics_cumulative,
            metrics_weekly=metrics_weekly,
            metrics_monthly=metrics_monthly,
        )

    def _compute_period_features(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        period_name: str,
    ) -> pd.DataFrame:
        """Compute clustering features for a given period (vectorized)."""
        # Filter returns to period
        mask = (self.returns_matrix.index >= start) & (self.returns_matrix.index <= end)
        period_returns = self.returns_matrix.loc[mask]

        if len(period_returns) == 0:
            return pd.DataFrame(index=self.returns_matrix.columns)

        # Count valid data points per column (vectorized)
        valid_counts = period_returns.notna().sum()

        # Initialize features DataFrame
        features = pd.DataFrame(index=self.returns_matrix.columns)

        # Mark columns with insufficient data
        insufficient_mask = valid_counts < self.min_data_points

        # Vectorized total return: (1 + r).prod() - 1
        # Use nanprod equivalent
        log_returns = np.log1p(period_returns.fillna(0))
        total_return = np.exp(log_returns.sum()) - 1
        features['return'] = total_return

        # Vectorized volatility (annualized)
        features['volatility'] = period_returns.std() * np.sqrt(252)

        # Vectorized Sharpe ratio
        mean_return = period_returns.mean() * 252
        with np.errstate(divide='ignore', invalid='ignore'):
            features['sharpe'] = np.where(
                features['volatility'] > 0,
                mean_return / features['volatility'],
                0.0
            )

        # Max drawdown (vectorized using cummax)
        equity = (1 + period_returns.fillna(0)).cumprod()
        running_max = equity.cummax()
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = (equity - running_max) / running_max
        features['max_drawdown'] = drawdown.min()

        # Calmar ratio (vectorized)
        with np.errstate(divide='ignore', invalid='ignore'):
            calmar = np.where(
                features['max_drawdown'] < 0,
                features['return'] / np.abs(features['max_drawdown']),
                np.where(features['return'] > 0, features['return'] * 10, 0)
            )
        features['calmar_ratio'] = np.clip(calmar, -10, 10)

        # Profit factor (vectorized)
        gains = period_returns.clip(lower=0).sum()
        losses = (-period_returns.clip(upper=0)).sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            profit_factor = np.where(
                losses > 0,
                gains / losses,
                np.where(gains > 0, gains * 10, 1.0)
            )
        features['profit_factor'] = np.clip(profit_factor, 0, 10)

        # Set NaN for insufficient data
        features.loc[insufficient_mask] = np.nan

        return features

    def _cluster_features(
        self,
        features: pd.DataFrame,
        horizon: str,
    ) -> tuple[pd.Series, dict]:
        """Cluster algorithms based on computed features."""
        # Filter out inactive algos (all NaN)
        active_mask = ~features.isna().all(axis=1)
        active_features = features[active_mask].copy()

        # Handle insufficient data
        if len(active_features) < self.n_clusters + 1:
            logger.debug(f"{horizon}: insufficient algos ({len(active_features)}) for clustering")
            labels = pd.Series('insufficient_data', index=features.index)
            return labels, {'silhouette': np.nan, 'n_clusters': 0}

        # Fill NaN with median for remaining algos
        active_features = active_features.fillna(active_features.median())

        # Replace inf values
        active_features = active_features.replace([np.inf, -np.inf], np.nan)
        active_features = active_features.fillna(active_features.median())

        try:
            clusterer = AlgoClusterer(
                method=self.method,
                n_clusters=min(self.n_clusters, len(active_features) - 1),
                features=self.CLUSTERING_FEATURES,
                scaler_type=self.scaler_type,
                random_state=self.random_state,
            )
            result = clusterer.fit(active_features)

            # Map labels to algo_ids
            labels = pd.Series(index=features.index, dtype=object)
            labels[~active_mask] = 'inactive'
            labels[active_mask] = result.labels.astype(str)

            metrics = {
                'silhouette': result.silhouette,
                'calinski_harabasz': result.calinski_harabasz,
                'davies_bouldin': result.davies_bouldin,
                'n_clusters': result.n_clusters,
            }

        except Exception as e:
            logger.warning(f"{horizon} clustering failed: {e}")
            labels = pd.Series('error', index=features.index)
            labels[~active_mask] = 'inactive'
            metrics = {'silhouette': np.nan, 'n_clusters': 0, 'error': str(e)}

        return labels, metrics

    def compare_all_methods(
        self,
        methods: Optional[list[ClusterMethod]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Compare all clustering methods across time horizons.

        Returns:
            Dict with comparison DataFrames per horizon
        """
        if methods is None:
            methods = list(ClusterMethod)

        logger.info(f"Comparing {len(methods)} clustering methods")

        # Sample a subset of weeks for comparison (every 4th week)
        sample_weeks = self.week_ends[::4]
        if len(sample_weeks) > 20:
            sample_weeks = sample_weeks[:20]

        comparisons = {'cumulative': [], 'weekly': [], 'monthly': []}

        for method in methods:
            logger.info(f"  Testing {method.value}...")

            original_method = self.method
            self.method = method

            method_metrics = {'cumulative': [], 'weekly': [], 'monthly': []}

            for week_end in sample_weeks:
                try:
                    result = self._cluster_single_week(week_end)
                    method_metrics['cumulative'].append(result.metrics_cumulative)
                    method_metrics['weekly'].append(result.metrics_weekly)
                    method_metrics['monthly'].append(result.metrics_monthly)
                except Exception as e:
                    logger.warning(f"Method {method.value} failed on {week_end}: {e}")

            self.method = original_method

            # Aggregate metrics per horizon
            for horizon in ['cumulative', 'weekly', 'monthly']:
                if method_metrics[horizon]:
                    avg_silhouette = np.nanmean([
                        m.get('silhouette', np.nan) for m in method_metrics[horizon]
                    ])
                    avg_calinski = np.nanmean([
                        m.get('calinski_harabasz', np.nan) for m in method_metrics[horizon]
                    ])
                    avg_davies = np.nanmean([
                        m.get('davies_bouldin', np.nan) for m in method_metrics[horizon]
                    ])
                    comparisons[horizon].append({
                        'method': method.value,
                        'avg_silhouette': avg_silhouette,
                        'avg_calinski_harabasz': avg_calinski,
                        'avg_davies_bouldin': avg_davies,
                    })

        return {
            horizon: pd.DataFrame(data).set_index('method')
            for horizon, data in comparisons.items()
            if data
        }

    def _select_best_methods(
        self,
        comparison: dict[str, pd.DataFrame],
    ) -> dict[str, ClusterMethod]:
        """Select best method per horizon based on silhouette score."""
        best = {}
        for horizon, df in comparison.items():
            if len(df) > 0 and 'avg_silhouette' in df.columns:
                best_method_name = df['avg_silhouette'].idxmax()
                best[horizon] = ClusterMethod(best_method_name)
            else:
                best[horizon] = ClusterMethod.KMEANS
        return best

    def save_results(
        self,
        output: TemporalClusteringOutput,
        save_path: str,
    ) -> None:
        """Save clustering results to disk."""
        from pathlib import Path

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save cluster history
        output.cluster_history.to_parquet(
            save_dir / 'cluster_history.parquet',
            index=False,
        )
        output.cluster_history.to_csv(
            save_dir / 'cluster_history.csv',
            index=False,
        )

        # Save method comparison
        for horizon, df in output.method_comparison.items():
            df.to_csv(save_dir / f'method_comparison_{horizon}.csv')

        # Save parameters
        import json
        with open(save_dir / 'params.json', 'w') as f:
            json.dump(output.params, f, indent=2)

        # Save best methods
        best_methods_str = {k: v.value for k, v in output.best_methods.items()}
        with open(save_dir / 'best_methods.json', 'w') as f:
            json.dump(best_methods_str, f, indent=2)

        logger.info(f"Results saved to {save_dir}")

    @staticmethod
    def load_results(load_path: str) -> TemporalClusteringOutput:
        """Load previously saved clustering results."""
        from pathlib import Path
        import json

        load_dir = Path(load_path)

        # Load cluster history
        cluster_history = pd.read_parquet(load_dir / 'cluster_history.parquet')

        # Load params
        with open(load_dir / 'params.json', 'r') as f:
            params = json.load(f)

        # Load best methods
        with open(load_dir / 'best_methods.json', 'r') as f:
            best_methods_str = json.load(f)
        best_methods = {k: ClusterMethod(v) for k, v in best_methods_str.items()}

        # Load method comparison
        method_comparison = {}
        for horizon in ['cumulative', 'weekly', 'monthly']:
            path = load_dir / f'method_comparison_{horizon}.csv'
            if path.exists():
                method_comparison[horizon] = pd.read_csv(path, index_col='method')

        return TemporalClusteringOutput(
            weekly_results=[],  # Not saved/loaded
            method_comparison=method_comparison,
            best_methods=best_methods,
            cluster_history=cluster_history,
            params=params,
        )

    def get_cluster_transitions(
        self,
        output: TemporalClusteringOutput,
        horizon: str = 'cumulative',
    ) -> pd.DataFrame:
        """
        Analyze cluster transitions over time.

        Returns DataFrame with transition counts between clusters.
        """
        df = output.cluster_history.copy()
        cluster_col = f'cluster_{horizon}'

        # Sort by algo and week
        df = df.sort_values(['algo_id', 'week_end'])

        # Compute transitions
        df['prev_cluster'] = df.groupby('algo_id')[cluster_col].shift(1)

        # Filter to actual transitions (exclude first week and inactive)
        transitions = df[
            (df['prev_cluster'].notna()) &
            (df[cluster_col] != 'inactive') &
            (df['prev_cluster'] != 'inactive')
        ]

        # Count transitions
        transition_counts = transitions.groupby(
            ['prev_cluster', cluster_col]
        ).size().unstack(fill_value=0)

        return transition_counts

    def get_cluster_stability(
        self,
        output: TemporalClusteringOutput,
        horizon: str = 'cumulative',
    ) -> pd.DataFrame:
        """
        Compute cluster stability per algorithm.

        Returns DataFrame with stability metrics per algo.
        """
        df = output.cluster_history.copy()
        cluster_col = f'cluster_{horizon}'

        stability = []
        for algo_id, group in df.groupby('algo_id'):
            active_clusters = group[group[cluster_col] != 'inactive'][cluster_col]

            if len(active_clusters) < 2:
                stability.append({
                    'algo_id': algo_id,
                    'n_weeks_active': len(active_clusters),
                    'n_cluster_changes': 0,
                    'stability_ratio': 1.0,
                    'dominant_cluster': active_clusters.mode().iloc[0] if len(active_clusters) > 0 else None,
                })
                continue

            # Count cluster changes
            n_changes = (active_clusters != active_clusters.shift()).sum() - 1
            stability_ratio = 1 - (n_changes / (len(active_clusters) - 1))

            stability.append({
                'algo_id': algo_id,
                'n_weeks_active': len(active_clusters),
                'n_cluster_changes': n_changes,
                'stability_ratio': stability_ratio,
                'dominant_cluster': active_clusters.mode().iloc[0],
            })

        return pd.DataFrame(stability)
