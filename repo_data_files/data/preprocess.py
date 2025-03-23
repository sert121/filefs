import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from PIL import Image
import cv2
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split

def setup_logging(log_dir: str) -> None:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "preprocess.log")),
            logging.StreamHandler()
        ]
    )

class DataPreprocessor:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        config: Dict[str, Any]
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        os.makedirs(output_dir, exist_ok=True)
        
    def preprocess_nerf_data(self) -> None:
        """Preprocess NeRF dataset."""
        logging.info("Preprocessing NeRF dataset...")
        
        # Load camera parameters
        with open(os.path.join(self.input_dir, "transforms.json"), "r") as f:
            transforms = json.load(f)
            
        # Process images
        images = []
        poses = []
        for frame in tqdm(transforms["frames"]):
            # Load image
            image_path = os.path.join(self.input_dir, frame["file_path"])
            image = Image.open(image_path)
            image = image.resize((
                self.config["nerf"]["data"]["image_size"],
                self.config["nerf"]["data"]["image_size"]
            ))
            images.append(np.array(image))
            
            # Extract camera pose
            pose = np.array(frame["transform_matrix"])
            poses.append(pose)
            
        # Convert to numpy arrays
        images = np.stack(images)
        poses = np.stack(poses)
        
        # Save processed data
        np.save(os.path.join(self.output_dir, "nerf_images.npy"), images)
        np.save(os.path.join(self.output_dir, "nerf_poses.npy"), poses)
        
    def preprocess_license_plate_data(self) -> None:
        """Preprocess license plate detection dataset."""
        logging.info("Preprocessing license plate dataset...")
        
        # Load annotations
        annotations = pd.read_csv(os.path.join(self.input_dir, "annotations.csv"))
        
        # Process images and annotations
        processed_data = []
        for _, row in tqdm(annotations.iterrows()):
            # Load image
            image_path = os.path.join(self.input_dir, row["image_path"])
            image = cv2.imread(image_path)
            image = cv2.resize(
                image,
                (
                    self.config["license_plate"]["data"]["image_size"],
                    self.config["license_plate"]["data"]["image_size"]
                )
            )
            
            # Process bounding box
            bbox = [
                row["x_min"],
                row["y_min"],
                row["x_max"],
                row["y_max"]
            ]
            
            processed_data.append({
                "image": image,
                "bbox": bbox,
                "class": row["class"]
            })
            
        # Split into train/val/test
        train_data, val_data = train_test_split(
            processed_data,
            test_size=0.2,
            random_state=self.config["global"]["seed"]
        )
        val_data, test_data = train_test_split(
            val_data,
            test_size=0.5,
            random_state=self.config["global"]["seed"]
        )
        
        # Save processed data
        torch.save(
            train_data,
            os.path.join(self.output_dir, "license_plate_train.pt")
        )
        torch.save(
            val_data,
            os.path.join(self.output_dir, "license_plate_val.pt")
        )
        torch.save(
            test_data,
            os.path.join(self.output_dir, "license_plate_test.pt")
        )
        
    def preprocess_clinical_data(self) -> None:
        """Preprocess clinical time series data."""
        logging.info("Preprocessing clinical dataset...")
        
        # Load raw data
        data = pd.read_csv(os.path.join(self.input_dir, "clinical_data.csv"))
        
        # Process features
        features = self.config["clinical"]["data"]["features"]
        sequence_length = self.config["clinical"]["data"]["sequence_length"]
        
        # Create sequences
        sequences = []
        labels = []
        for patient_id in tqdm(data["patient_id"].unique()):
            patient_data = data[data["patient_id"] == patient_id]
            
            # Create sequences of specified length
            for i in range(len(patient_data) - sequence_length):
                seq = patient_data[features].iloc[i:i+sequence_length].values
                label = patient_data["mortality"].iloc[i+sequence_length]
                
                sequences.append(seq)
                labels.append(label)
                
        # Convert to numpy arrays
        sequences = np.stack(sequences)
        labels = np.array(labels)
        
        # Split into train/val/test
        train_seq, val_seq, train_labels, val_labels = train_test_split(
            sequences,
            labels,
            test_size=0.2,
            random_state=self.config["global"]["seed"]
        )
        val_seq, test_seq, val_labels, test_labels = train_test_split(
            val_seq,
            val_labels,
            test_size=0.5,
            random_state=self.config["global"]["seed"]
        )
        
        # Save processed data
        np.save(os.path.join(self.output_dir, "clinical_train.npy"), train_seq)
        np.save(os.path.join(self.output_dir, "clinical_val.npy"), val_seq)
        np.save(os.path.join(self.output_dir, "clinical_test.npy"), test_seq)
        np.save(os.path.join(self.output_dir, "clinical_train_labels.npy"), train_labels)
        np.save(os.path.join(self.output_dir, "clinical_val_labels.npy"), val_labels)
        np.save(os.path.join(self.output_dir, "clinical_test_labels.npy"), test_labels)
        
    def preprocess_legal_data(self) -> None:
        """Preprocess legal text data."""
        logging.info("Preprocessing legal dataset...")
        
        # Load raw data
        with open(os.path.join(self.input_dir, "legal_cases.json"), "r") as f:
            data = json.load(f)
            
        # Process texts
        texts = []
        labels = []
        for case in tqdm(data):
            # Clean text
            text = case["text"]
            if self.config["legal_bert"]["data"]["preprocessing"]["remove_special_chars"]:
                text = "".join(c for c in text if c.isalnum() or c.isspace())
            if self.config["legal_bert"]["data"]["preprocessing"]["normalize_whitespace"]:
                text = " ".join(text.split())
                
            texts.append(text)
            labels.append(case["label"])
            
        # Split into train/val/test
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=self.config["global"]["seed"]
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            val_texts,
            val_labels,
            test_size=0.5,
            random_state=self.config["global"]["seed"]
        )
        
        # Save processed data
        torch.save(
            {"texts": train_texts, "labels": train_labels},
            os.path.join(self.output_dir, "legal_train.pt")
        )
        torch.save(
            {"texts": val_texts, "labels": val_labels},
            os.path.join(self.output_dir, "legal_val.pt")
        )
        torch.save(
            {"texts": test_texts, "labels": test_labels},
            os.path.join(self.output_dir, "legal_test.pt")
        )
        
    def preprocess_diffusion_data(self) -> None:
        """Preprocess diffusion model dataset."""
        logging.info("Preprocessing diffusion dataset...")
        
        # Load image paths and labels
        with open(os.path.join(self.input_dir, "image_paths.json"), "r") as f:
            data = json.load(f)
            
        # Process images
        processed_data = []
        for item in tqdm(data):
            # Load and resize image
            image_path = os.path.join(self.input_dir, item["path"])
            image = Image.open(image_path)
            image = image.resize((
                self.config["diffusion"]["data"]["image_size"],
                self.config["diffusion"]["data"]["image_size"]
            ))
            
            processed_data.append({
                "image": np.array(image),
                "label": item["label"]
            })
            
        # Split into train/val/test
        train_data, val_data = train_test_split(
            processed_data,
            test_size=0.2,
            random_state=self.config["global"]["seed"]
        )
        val_data, test_data = train_test_split(
            val_data,
            test_size=0.5,
            random_state=self.config["global"]["seed"]
        )
        
        # Save processed data
        torch.save(
            train_data,
            os.path.join(self.output_dir, "diffusion_train.pt")
        )
        torch.save(
            val_data,
            os.path.join(self.output_dir, "diffusion_val.pt")
        )
        torch.save(
            test_data,
            os.path.join(self.output_dir, "diffusion_test.pt")
        )

def main():
    # Load configuration
    with open("config/training_config.yml", "r") as f:
        config = yaml.safe_load(f)
        
    # Setup logging
    setup_logging(config["logging"]["log_dir"])
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        input_dir="data/raw",
        output_dir="data/processed",
        config=config
    )
    
    # Preprocess each dataset
    preprocessor.preprocess_nerf_data()
    preprocessor.preprocess_license_plate_data()
    preprocessor.preprocess_clinical_data()
    preprocessor.preprocess_legal_data()
    preprocessor.preprocess_diffusion_data()
    
    logging.info("Data preprocessing completed!")

if __name__ == "__main__":
    main() 