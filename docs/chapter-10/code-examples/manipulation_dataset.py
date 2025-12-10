#!/usr/bin/env python3

"""
VLA Manipulation Dataset

This script demonstrates a dataset for Vision-Language-Action manipulation training,
including data loading, preprocessing, and training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
import os


class ManipulationDataset(Dataset):
    """Dataset for VLA manipulation training"""
    def __init__(self, data_dir, transforms=None, max_samples=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.data_samples = []

        # Load manipulation dataset metadata
        metadata_path = os.path.join(data_dir, "manipulation_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.data_samples = metadata['manipulation_samples']
        else:
            # For demo purposes, create synthetic manipulation data
            self.data_samples = self._create_synthetic_manipulation_data(500)

        if max_samples:
            self.data_samples = self.data_samples[:max_samples]

        print(f"Loaded {len(self.data_samples)} manipulation samples from {data_dir}")

    def _create_synthetic_manipulation_data(self, num_samples):
        """Create synthetic manipulation training data for demonstration"""
        samples = []
        manipulation_actions = [
            "grasp_object", "place_object", "push_object",
            "pull_object", "lift_object", "rotate_object"
        ]

        for i in range(num_samples):
            action_type = np.random.choice(manipulation_actions)
            sample = {
                'image_path': f"manipulation_image_{i:04d}.jpg",
                'language_instruction': f"Please {action_type.replace('_', ' ')} the object",
                'grasp_pose': np.random.randn(6).tolist(),  # 6-DOF grasp pose
                'trajectory': np.random.randn(20, 7).tolist(),  # 20 timesteps, 7-DOF trajectory
                'force_control': np.random.randn(6).tolist(),  # 6-DOF force/torque
                'object_properties': {
                    'size': np.random.uniform(0.05, 0.2),
                    'weight': np.random.uniform(0.1, 2.0),
                    'shape': np.random.choice(['cylinder', 'box', 'sphere'])
                }
            }
            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        # Load and preprocess image (simulated)
        image = self._load_image(sample['image_path'])
        if self.transforms:
            image = self.transforms(image)

        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()

        # Get text description
        text = sample['language_instruction']

        # Get manipulation targets
        grasp_pose = torch.tensor(sample['grasp_pose'], dtype=torch.float)
        trajectory = torch.tensor(sample['trajectory'], dtype=torch.float)
        force_control = torch.tensor(sample['force_control'], dtype=torch.float)

        return {
            'image': image_tensor,
            'text': text,
            'grasp_pose': grasp_pose,
            'trajectory': trajectory,
            'force_control': force_control,
            'object_properties': sample['object_properties']
        }

    def _load_image(self, path):
        """Load an image (simulated for this example)"""
        # In practice, this would load an actual image from disk
        # For demonstration, we'll create a dummy image
        return np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8) / 255.0


class SimpleManipulationModel(nn.Module):
    """Simple manipulation model for demonstration"""
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Linear(3*224*224, 512)
        self.language_encoder = nn.Linear(512, 512)
        self.fusion = nn.Linear(512*2, 256)

        # Output heads for different manipulation aspects
        self.grasp_head = nn.Linear(256, 6)  # 6-DOF grasp pose
        self.trajectory_head = nn.Linear(256, 140)  # 20 timesteps * 7 DOF
        self.force_head = nn.Linear(256, 6)  # 6-DOF force/torque

    def forward(self, images, text_features):
        batch_size = images.size(0)

        # Process vision
        vision_flat = images.view(batch_size, -1)
        vision_encoded = self.vision_encoder(vision_flat)

        # Process language (using the text_features directly)
        language_encoded = self.language_encoder(text_features)

        # Fuse multimodal features
        fused = self.fusion(torch.cat([vision_encoded, language_encoded], dim=-1))

        # Generate manipulation outputs
        grasp_pose = self.grasp_head(fused)
        trajectory = self.trajectory_head(fused)
        force_control = self.force_head(fused)

        return {
            'grasp_pose': grasp_pose,
            'trajectory': trajectory.view(batch_size, 20, 7),  # Reshape trajectory
            'force_control': force_control
        }


class ManipulationTrainer:
    """Trainer for VLA manipulation models"""
    def __init__(self, model, train_dataset, val_dataset, batch_size=16, learning_rate=1e-4):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Separate losses for different manipulation outputs
        self.grasp_loss_fn = nn.MSELoss()
        self.trajectory_loss_fn = nn.MSELoss()
        self.force_loss_fn = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        grasp_loss_total = 0
        trajectory_loss_total = 0
        force_loss_total = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            texts = batch['text']
            grasp_poses = batch['grasp_pose'].to(self.device)
            trajectories = batch['trajectory'].to(self.device)
            force_controls = batch['force_control'].to(self.device)

            # Encode text descriptions (simplified for demo)
            text_features = torch.randn(len(texts), 512).to(self.device)  # Dummy encoding

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, text_features)

            # Calculate losses for different manipulation outputs
            grasp_loss = self.grasp_loss_fn(outputs['grasp_pose'], grasp_poses)
            trajectory_loss = self.trajectory_loss_fn(outputs['trajectory'], trajectories)
            force_loss = self.force_loss_fn(outputs['force_control'], force_controls)

            # Combined loss
            total_batch_loss = grasp_loss + trajectory_loss + force_loss

            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            grasp_loss_total += grasp_loss.item()
            trajectory_loss_total += trajectory_loss.item()
            force_loss_total += force_loss.item()

        return {
            'total': total_loss / len(self.train_loader),
            'grasp': grasp_loss_total / len(self.train_loader),
            'trajectory': trajectory_loss_total / len(self.train_loader),
            'force': force_loss_total / len(self.train_loader)
        }

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        grasp_loss_total = 0
        trajectory_loss_total = 0
        force_loss_total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                texts = batch['text']
                grasp_poses = batch['grasp_pose'].to(self.device)
                trajectories = batch['trajectory'].to(self.device)
                force_controls = batch['force_control'].to(self.device)

                # Encode text descriptions (simplified for demo)
                text_features = torch.randn(len(texts), 512).to(self.device)  # Dummy encoding

                outputs = self.model(images, text_features)

                grasp_loss = self.grasp_loss_fn(outputs['grasp_pose'], grasp_poses)
                trajectory_loss = self.trajectory_loss_fn(outputs['trajectory'], trajectories)
                force_loss = self.force_loss_fn(outputs['force_control'], force_controls)

                total_batch_loss = grasp_loss + trajectory_loss + force_loss

                total_loss += total_batch_loss.item()
                grasp_loss_total += grasp_loss.item()
                trajectory_loss_total += trajectory_loss.item()
                force_loss_total += force_loss.item()

        return {
            'total': total_loss / len(self.val_loader),
            'grasp': grasp_loss_total / len(self.val_loader),
            'trajectory': trajectory_loss_total / len(self.val_loader),
            'force': force_loss_total / len(self.val_loader)
        }

    def train(self, num_epochs):
        """Train the VLA manipulation model"""
        print("Starting VLA manipulation model training...")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 30)

            train_losses = self.train_epoch()
            val_losses = self.validate()

            print(f"Train - Total: {train_losses['total']:.4f}, Grasp: {train_losses['grasp']:.4f}, "
                  f"Trajectory: {train_losses['trajectory']:.4f}, Force: {train_losses['force']:.4f}")
            print(f"Val - Total: {val_losses['total']:.4f}, Grasp: {val_losses['grasp']:.4f}, "
                  f"Trajectory: {val_losses['trajectory']:.4f}, Force: {val_losses['force']:.4f}")

        print("Manipulation training completed!")


def main():
    """Main function to demonstrate manipulation dataset and training"""
    print("Initializing VLA Manipulation Dataset and Training...")

    # Create synthetic manipulation datasets for demonstration
    train_dataset = ManipulationDataset(data_dir="./dummy_manipulation_data", max_samples=400)
    val_dataset = ManipulationDataset(data_dir="./dummy_manipulation_data", max_samples=100)

    # Initialize the simple manipulation model
    model = SimpleManipulationModel()

    # Initialize trainer
    trainer = ManipulationTrainer(model, train_dataset, val_dataset, batch_size=8, learning_rate=1e-4)

    # Train the model
    try:
        trainer.train(num_epochs=5)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")

    # Demonstrate data loading
    print("\n--- Data Loading Example ---")
    sample_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    sample_batch = next(iter(sample_loader))

    print(f"Image batch shape: {sample_batch['image'].shape}")
    print(f"Text batch: {sample_batch['text'][:2]}")
    print(f"Grasp pose batch shape: {sample_batch['grasp_pose'].shape}")
    print(f"Trajectory batch shape: {sample_batch['trajectory'].shape}")
    print(f"Object properties: {sample_batch['object_properties']}")


if __name__ == "__main__":
    main()