import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Optional


class PhoBERTClassifier(nn.Module):
    """
    PhoBERT-based text classifier for Vietnamese news classification
    
    Kiến trúc:
    - PhoBERT (pre-trained): Trích xuất đặc trưng từ văn bản tiếng Việt
    - Fully Connected Layer: Lớp phân loại với dropout để tránh overfitting
    - Sigmoid/Softmax: Activation function cho output
    """
    
    def __init__(
        self, 
        num_classes: int,
        phobert_model: str = 'vinai/phobert-base',
        dropout_rate: float = 0.3,
        hidden_size: int = 768,
        freeze_phobert: bool = False
    ):
        """
        Khởi tạo PhoBERT Classifier
        
        Args:
            num_classes: Số lượng class cần phân loại
            phobert_model: Tên model PhoBERT từ HuggingFace
            dropout_rate: Tỉ lệ dropout để tránh overfitting
            hidden_size: Kích thước hidden layer từ PhoBERT (768 cho base, 1024 cho large)
            freeze_phobert: Có đóng băng trọng số PhoBERT hay không
        """
        super(PhoBERTClassifier, self).__init__()
        
        # Load pre-trained PhoBERT model
        self.phobert = AutoModel.from_pretrained(phobert_model)
        
        # Freeze PhoBERT parameters if needed (chỉ train FC layer)
        if freeze_phobert:
            for param in self.phobert.parameters():
                param.requires_grad = False
        
        # Fully Connected Layer cho phân loại
        # PhoBERT -> FC Layer -> Output
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        self.num_classes = num_classes
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs từ tokenizer (batch_size, seq_length)
            attention_mask: Mask để PhoBERT biết token nào cần attention (batch_size, seq_length)
            
        Returns:
            logits: Output scores cho mỗi class (batch_size, num_classes)
        """
        # 1. PhoBERT encoding
        # outputs[0]: last hidden state (batch_size, seq_length, hidden_size)
        # outputs[1]: pooled output - [CLS] token representation (batch_size, hidden_size)
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 2. Lấy representation từ [CLS] token (token đầu tiên)
        # Theo Figure 2: final layer representation matching to the token
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        
        # 3. Dropout để regularization
        pooled_output = self.dropout(pooled_output)
        
        # 4. Fully Connected Layer để phân loại
        logits = self.fc(pooled_output)  # (batch_size, num_classes)
        
        return logits


class PhoBERTTextPreprocessor:
    """
    Tokenization cho văn bản tiếng Việt đã được tiền xử lý
    Tương ứng với bước "Encode" trong Figure 2
    
    Lưu ý: Class này giả định dữ liệu đã được tiền xử lý trước đó
    """
    
    def __init__(
        self, 
        phobert_model: str = 'vinai/phobert-base',
        max_length: int = 128
    ):
        """
        Khởi tạo preprocessor
        
        Args:
            phobert_model: Tên model PhoBERT
            max_length: Độ dài tối đa của sequence
                       - 128: Đủ cho title bài báo
                       
        """
        self.tokenizer = AutoTokenizer.from_pretrained(phobert_model)
        self.max_length = max_length
    
    def encode_texts(
        self, 
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> dict:
        """
        Encode danh sách văn bản thành tokens
        Tương ứng với bước "Encode" trong Figure 2
        
        Args:
            texts: Danh sách văn bản đã được tiền xử lý cần encode
            padding: Có padding các sequence về cùng độ dài không
            truncation: Có cắt ngắn sequence vượt quá max_length không
            return_tensors: Format output ('pt' cho PyTorch, 'tf' cho TensorFlow)
            
        Returns:
            Dictionary chứa input_ids, attention_mask
        """
        # Tokenize và encode văn bản đã được tiền xử lý
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        return encoding


class PhoBERTTrainer:
    """
    Trainer để huấn luyện PhoBERT Classifier
    Sử dụng Binary Cross-Entropy Loss như trong công thức (1) của Figure 2
    """
    
    def __init__(
        self,
        model: PhoBERTClassifier,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Khởi tạo trainer
        
        Args:
            model: PhoBERT classifier model
            device: Device để train (cuda/cpu)
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        # Binary Cross-Entropy với Sigmoid activation
        if model.num_classes == 2:
            # Binary classification
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            # Multi-class classification
            self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(
        self,
        train_loader,
        optimizer,
        scheduler=None
    ) -> Tuple[float, float]:
        """
        Train model trong một epoch
        
        Args:
            train_loader: DataLoader cho training data
            optimizer: Optimizer (Adam, AdamW, etc.)
            scheduler: Learning rate scheduler (optional)
            
        Returns:
            avg_loss: Loss trung bình
            avg_acc: Accuracy trung bình
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            if self.model.num_classes == 2:
                loss = self.criterion(logits.squeeze(), labels.float())
                predictions = (torch.sigmoid(logits.squeeze()) > 0.5).long()
            else:
                loss = self.criterion(logits, labels)
                predictions = torch.argmax(logits, dim=1)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def evaluate(
        self,
        val_loader
    ) -> Tuple[float, float]:
        """
        Evaluate model trên validation set
        
        Args:
            val_loader: DataLoader cho validation data
            
        Returns:
            avg_loss: Loss trung bình
            avg_acc: Accuracy trung bình
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                if self.model.num_classes == 2:
                    loss = self.criterion(logits.squeeze(), labels.float())
                    predictions = (torch.sigmoid(logits.squeeze()) > 0.5).long()
                else:
                    loss = self.criterion(logits, labels)
                    predictions = torch.argmax(logits, dim=1)
                
                # Statistics
                total_loss += loss.item()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def predict(
        self,
        texts: List[str],
        preprocessor: PhoBERTTextPreprocessor,
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Predict labels cho danh sách văn bản
        
        Args:
            texts: Danh sách văn bản cần predict
            preprocessor: PhoBERTTextPreprocessor để encode texts
            batch_size: Batch size cho prediction
            
        Returns:
            predictions: Mảng predicted labels
        """
        self.model.eval()
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Encode texts
            encoding = preprocessor.encode_texts(batch_texts)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                
                if self.model.num_classes == 2:
                    predictions = (torch.sigmoid(logits.squeeze()) > 0.5).long()
                else:
                    predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
        
        return np.array(all_predictions)


# Example usage
if __name__ == "__main__":
    """
    Ví dụ sử dụng PhoBERT cho phân loại tin tức tiếng Việt
    """
    
    # 1. Khởi tạo preprocessor
    print("=== Bước 1: Khởi tạo Preprocessor ===")
    preprocessor = PhoBERTTextPreprocessor(
        phobert_model='vinai/phobert-base',
        max_length=128
    )
    
    # 2. Khởi tạo model
    print("=== Bước 2: Khởi tạo PhoBERT Classifier ===")
    num_classes = 13  # Ví dụ: 13 categories tin tức
    model = PhoBERTClassifier(
        num_classes=num_classes,
        phobert_model='vinai/phobert-base',
        dropout_rate=0.3,
        hidden_size=768,
        freeze_phobert=False  # False: train cả PhoBERT, True: chỉ train FC layer
    )
    
    print(f"Model architecture:\n{model}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 3. Ví dụ encode text
    print("\n=== Bước 3: Ví dụ Preprocessing và Encoding ===")
    sample_texts = [
        "Việt Nam đã giành chiến thắng trong trận đấu bóng đá quan trọng.",
        "Chính phủ công bố chính sách mới về phát triển kinh tế số."
    ]
    
    encoding = preprocessor.encode_texts(sample_texts)
    print(f"Input IDs shape: {encoding['input_ids'].shape}")
    print(f"Attention Mask shape: {encoding['attention_mask'].shape}")
    
    # 4. Forward pass demo
    print("\n=== Bước 4: Ví dụ Forward Pass ===")
    model.eval()
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        print(f"Output logits shape: {logits.shape}")
        print(f"Predicted classes: {torch.argmax(logits, dim=1)}")
    
    print("\n=== Setup hoàn tất! ===")
    print("Bạn có thể sử dụng PhoBERTTrainer để train model trên dataset của bạn.")
    
    # ========================================
    # DEMO: TRAINING VỚI DATASET THỰC
    # ========================================
    # print("\n" + "="*60)
    # print("DEMO: HUẤN LUYỆN MODEL VỚI DATASET UIT-ViON")
    # print("="*60)
    
    # import pandas as pd
    # from torch.utils.data import Dataset, DataLoader
    # from sklearn.preprocessing import LabelEncoder
    
    # # ======================
    # # 1. TẠO CUSTOM DATASET
    # # ======================
    # class VietnameseNewsDataset(Dataset):
    #     """
    #     Custom Dataset cho Vietnamese News Classification
    #     """
    #     def __init__(
    #         self, 
    #         csv_file: str,
    #         preprocessor: PhoBERTTextPreprocessor,
    #         label_encoder: Optional[LabelEncoder] = None
    #     ):
    #         """
    #         Args:
    #             csv_file: Path đến file CSV (có columns: title, label)
    #             preprocessor: PhoBERTTextPreprocessor để encode texts
    #             label_encoder: LabelEncoder để encode labels (None sẽ tự tạo mới)
    #         """
    #         self.data = pd.read_csv(csv_file)
    #         self.preprocessor = preprocessor
            
    #         # Encode labels
    #         if label_encoder is None:
    #             self.label_encoder = LabelEncoder()
    #             self.labels = self.label_encoder.fit_transform(self.data['label'])
    #         else:
    #             self.label_encoder = label_encoder
    #             self.labels = self.label_encoder.transform(self.data['label'])
            
    #         self.texts = self.data['title'].tolist()
            
    #     def __len__(self):
    #         return len(self.texts)
        
    #     def __getitem__(self, idx):
    #         text = self.texts[idx]
    #         label = self.labels[idx]
            
    #         # Encode single text
    #         encoding = self.preprocessor.encode_texts(
    #             [text],
    #             padding='max_length',
    #             truncation=True,
    #             return_tensors='pt'
    #         )
            
    #         return {
    #             'input_ids': encoding['input_ids'].squeeze(0),
    #             'attention_mask': encoding['attention_mask'].squeeze(0),
    #             'labels': torch.tensor(label, dtype=torch.long)
    #         }
    
    # print("\n=== Bước 5: Load Dataset ===")
    
    # # Khởi tạo preprocessor và model mới
    # preprocessor_train = PhoBERTTextPreprocessor(
    #     phobert_model='vinai/phobert-base',
    #     max_length=128
    # )
    
    # # Load training dataset
    # train_dataset = VietnameseNewsDataset(
    #     csv_file='data/preprocess/UIT-ViON_train_preprocessed.csv',
    #     preprocessor=preprocessor_train
    # )
    
    # # Load validation dataset (sử dụng label_encoder từ train)
    # val_dataset = VietnameseNewsDataset(
    #     csv_file='data/preprocess/UIT-ViON_dev_preprocessed.csv',
    #     preprocessor=preprocessor_train,
    #     label_encoder=train_dataset.label_encoder
    # )
    
    # print(f"Training samples: {len(train_dataset):,}")
    # print(f"Validation samples: {len(val_dataset):,}")
    # print(f"Number of classes: {len(train_dataset.label_encoder.classes_)}")
    # print(f"Classes: {train_dataset.label_encoder.classes_}")
    
    # # =======================
    # # 2. TẠO DATA LOADERS
    # # =======================
    # print("\n=== Bước 6: Tạo DataLoaders ===")
    
    # BATCH_SIZE = 16  # Điều chỉnh tùy theo GPU memory
    # NUM_WORKERS = 2
    
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=True if torch.cuda.is_available() else False
    # )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=True if torch.cuda.is_available() else False
    # )
    
    # print(f"Training batches: {len(train_loader)}")
    # print(f"Validation batches: {len(val_loader)}")
    
    # # ===========================
    # # 3. KHỞI TẠO MODEL & TRAINER
    # # ===========================
    # print("\n=== Bước 7: Khởi tạo Model & Trainer ===")
    
    # # Khởi tạo model
    # num_classes = len(train_dataset.label_encoder.classes_)
    # model_train = PhoBERTClassifier(
    #     num_classes=num_classes,
    #     phobert_model='vinai/phobert-base',
    #     dropout_rate=0.3,
    #     hidden_size=768,
    #     freeze_phobert=False  # Train cả PhoBERT
    # )
    
    # # Khởi tạo trainer
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")
    
    # trainer = PhoBERTTrainer(
    #     model=model_train,
    #     device=device
    # )
    
    # # ========================
    # # 4. SETUP OPTIMIZER & LR
    # # ========================
    # print("\n=== Bước 8: Setup Optimizer & Learning Rate ===")
    
    # from torch.optim import AdamW
    # from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
    
    # # Optimizer
    # LEARNING_RATE = 2e-5  # Learning rate nhỏ cho fine-tuning PhoBERT
    # WEIGHT_DECAY = 0.01
    
    # optimizer = AdamW(
    #     model_train.parameters(),
    #     lr=LEARNING_RATE,
    #     weight_decay=WEIGHT_DECAY
    # )
    
    # # Learning rate scheduler
    # NUM_EPOCHS = 5
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=NUM_EPOCHS * len(train_loader)
    # )
    
    # print(f"Learning rate: {LEARNING_RATE}")
    # print(f"Weight decay: {WEIGHT_DECAY}")
    # print(f"Number of epochs: {NUM_EPOCHS}")
    
    # # ====================
    # # 5. TRAINING LOOP
    # # ====================
    # print("\n=== Bước 9: Bắt đầu Training ===")
    # print("-" * 60)
    
    # best_val_acc = 0.0
    # history = {
    #     'train_loss': [],
    #     'train_acc': [],
    #     'val_loss': [],
    #     'val_acc': []
    # }
    
    # for epoch in range(NUM_EPOCHS):
    #     print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    #     print("-" * 40)
        
    #     # Training
    #     train_loss, train_acc = trainer.train_epoch(
    #         train_loader=train_loader,
    #         optimizer=optimizer,
    #         scheduler=scheduler
    #     )
        
    #     # Validation
    #     val_loss, val_acc = trainer.evaluate(val_loader=val_loader)
        
    #     # Save history
    #     history['train_loss'].append(train_loss)
    #     history['train_acc'].append(train_acc)
    #     history['val_loss'].append(val_loss)
    #     history['val_acc'].append(val_acc)
        
    #     # Print results
    #     print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    #     print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
    #     # Save best model
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model_train.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'val_acc': val_acc,
    #             'label_encoder': train_dataset.label_encoder
    #         }, 'best_phobert_model.pth')
    #         print(f"✓ Saved best model with validation accuracy: {val_acc:.4f}")
    
    # print("\n" + "="*60)
    # print("TRAINING HOÀN TẤT!")
    # print("="*60)
    # print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    # print(f"Model đã được lưu tại: best_phobert_model.pth")
    
    # # ========================
    # # 6. TEST PREDICTION
    # # ========================
    # print("\n=== Bước 10: Test Prediction ===")
    
    # # Load best model
    # checkpoint = torch.load('best_phobert_model.pth')
    # model_train.load_state_dict(checkpoint['model_state_dict'])
    
    # # Test với một vài samples
    # test_texts = [
    #     "ronaldo ghi 2 bàn_thắng trận chung_kết",
    #     "chính_phủ ban_hành luật mới về thuế",
    #     "iphone 15 ra_mắt tính_năng mới"
    # ]
    
    # print("\nTest predictions:")
    # predictions = trainer.predict(
    #     texts=test_texts,
    #     preprocessor=preprocessor_train,
    #     batch_size=8
    # )
    
    # for text, pred_idx in zip(test_texts, predictions):
    #     pred_label = train_dataset.label_encoder.inverse_transform([pred_idx])[0]
    #     print(f"Text: {text}")
    #     print(f"Predicted label: {pred_label}\n")
    
    # print("\n=== HOÀN TẤT! ===")
    # print("""
    # Để sử dụng model đã train:
    
    # 1. Load model:
    #    checkpoint = torch.load('best_phobert_model.pth')
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #    label_encoder = checkpoint['label_encoder']
    
    # 2. Predict:
    #    predictions = trainer.predict(texts, preprocessor)
    #    labels = label_encoder.inverse_transform(predictions)
    # """)
