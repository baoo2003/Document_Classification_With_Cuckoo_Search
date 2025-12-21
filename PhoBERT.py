import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Optional, Sequence, Union
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm.auto import tqdm
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

class PhoBERTClassifier(nn.Module):
    """
    PhoBERT-based text classifier for Vietnamese news classification
    
    Kiến trúc (Custom Architecture):
    - PhoBERT (pre-trained): Trích xuất đặc trưng (last_hidden_state của token [CLS])
    - Dropout: Giảm overfitting
    - Linear: Classification Head (hidden_size -> num_classes)
    """
    
    def __init__(
        self, 
        num_classes: int,
        phobert_model: str = 'vinai/phobert-base',
        local_dir: str = 'models/phobert-base',
        dropout_rate: float = 0.1,
        hidden_size: int = 768,
        freeze_phobert: bool = False
    ):
        """
        Khởi tạo PhoBERT Classifier
        
        Args:
            num_classes: Số lượng class cần phân loại
            phobert_model: Tên model PhoBERT từ HuggingFace
            dropout_rate: Tỉ lệ dropout
            hidden_size: Kích thước hidden layer của PhoBERT (768 cho base)
            freeze_phobert: Có đóng băng trọng số PhoBERT hay không
        """
        super(PhoBERTClassifier, self).__init__()
        
        # Load pre-trained PhoBERT model (Base model)
        config_path = os.path.join(local_dir, 'config.json')
        if os.path.exists(config_path):
            print(f">> Loading PhoBERT model from local: {local_dir}")
            self.phobert = AutoModel.from_pretrained(
                local_dir, 
                local_files_only=True
            )
        else:
            print(f">> Downloading PhoBERT model from HuggingFace: {phobert_model}")
            os.makedirs(local_dir, exist_ok=True)
            self.phobert = AutoModel.from_pretrained(phobert_model)
            self.phobert.save_pretrained(local_dir)
        
        # Custom Classification Head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Freeze PhoBERT parameters if needed
        if freeze_phobert:
            for param in self.phobert.parameters():
                param.requires_grad = False
        
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
        # Get PhoBERT outputs
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token representation (first token)
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classifier
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        
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
        local_dir: str = 'models/phobert-base',
        max_length: int = 128
    ):
        """
        Khởi tạo preprocessor
        
        Args:
            phobert_model: Tên model PhoBERT
            max_length: Độ dài tối đa của sequence
                       - 128: Đủ cho title bài báo
                       
        """
        # Kiểm tra xem local directory có file tokenizer_config.json không
        tokenizer_config_path = os.path.join(local_dir, 'tokenizer_config.json')
        if os.path.exists(tokenizer_config_path):
            print(f">> Loading PhoBERT tokenizer from local: {local_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        else:
            print(f">> Downloading PhoBERT tokenizer from HuggingFace: {phobert_model}")
            os.makedirs(local_dir, exist_ok=True)
            self.tokenizer = AutoTokenizer.from_pretrained(phobert_model)
            self.tokenizer.save_pretrained(local_dir)
        self.max_length = max_length
    
    def encode_texts(
        self, 
        texts: List[str],
        text_pairs: Optional[List[Optional[str]]] = None,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> dict:
        """
        Encode danh sách văn bản thành tokens.
        Hỗ trợ cả input 1 câu (texts) và input 2 câu (texts + text_pairs).
        
        Args:
            texts: Danh sách văn bản đã được tiền xử lý cần encode
            text_pairs: Danh sách văn bản thứ hai (ví dụ: description). Nếu None -> encode 1 câu.
            padding: Có padding các sequence về cùng độ dài không
            truncation: Có cắt ngắn sequence vượt quá max_length không
            return_tensors: Format output ('pt' cho PyTorch, 'tf' cho TensorFlow)
            
        Returns:
            Dictionary chứa input_ids, attention_mask
        """
        # Tokenize và encode văn bản đã được tiền xử lý
        # Với PhoBERT (RoBERTa-like), text_pair sẽ tự động chèn đúng special tokens.
        encoding = self.tokenizer(
            texts,
            text_pairs,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            return_overflowing_tokens=False,
        )
        
        return encoding

    def encode_title_description(
        self,
        titles: List[str],
        descriptions: List[Optional[str]],
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> dict:
        """Convenience wrapper cho input dạng (title, description)."""
        
        return self.encode_texts(
            titles,
            text_pairs=descriptions,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )


class PhoBERTTrainer:
    """
    Trainer để huấn luyện PhoBERT Classifier
    Sử dụng Cross-Entropy Loss (Standard cho Multi-class Classification)
    """
    
    def __init__(
        self,
        model: PhoBERTClassifier,
        device: str = 'cuda',
        label_smoothing: float = 0.0
    ):
        """
        Khởi tạo trainer
        
        Args:
            model: PhoBERT classifier model
            device: Device để train (mặc định là cuda)
            label_smoothing: Chỉ số làm mượt nhãn (0.0 - 1.0)
        """
        self.model = model.to(device)
        self.device = device
        
        # Tối ưu hóa cho CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Sử dụng CrossEntropyLoss cho bài toán Multi-class (13 classes)
        # Hàm này đã tích hợp sẵn LogSoftmax + NLLLoss
        # Tự động xử lý việc tính Softmax bên trong
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def train_epoch(
        self,
        train_loader,
        optimizer,
        scheduler=None,
        accumulation_steps: int = 1,
        use_mixed_precision: bool = True,
        max_grad_norm: float = 1.0,
        show_progress: bool = True
    ) -> Tuple[float, float]:
        """
        Train model trong một epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision scaler
        scaler = GradScaler() if use_mixed_precision else None
        
        progress_bar = tqdm(train_loader, desc="Training", ncols=100) if show_progress else train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Forward pass
            if scaler:
                with autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
            
            # Gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Statistics
            total_loss += loss.item() * accumulation_steps
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            if show_progress:
                current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{total_loss / (batch_idx + 1):.4f}',
                    'acc': f'{correct / total:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc

    def evaluate(
        self,
        val_loader,
        show_progress: bool = False,
        use_mixed_precision: bool = True
    ) -> Tuple[float, float]:
        """
        Evaluate model trên validation set
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(val_loader, desc="Validating", ncols=100) if show_progress else val_loader
    
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass
                if use_mixed_precision:
                    with autocast():
                        logits = self.model(input_ids, attention_mask)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                # Calculate predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Statistics
                total_loss += loss.item()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                if show_progress:
                    progress_bar.set_postfix({
                        'loss': f'{total_loss / (total / labels.size(0)):.4f}',
                        'acc': f'{correct / total:.4f}'
                    })
    
        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def predict_from_loader(
        self,
        loader,
        use_mixed_precision: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels cho toàn bộ dataset trong loader
        """
        self.model.eval()
        all_preds = []
        all_labels = []
            
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting", leave=False):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                if use_mixed_precision:
                    with autocast():
                        logits = self.model(input_ids, attention_mask)
                else:
                    logits = self.model(input_ids, attention_mask)
                
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        return np.array(all_preds), np.array(all_labels)

    def predict(
        self,
        texts: Union[List[str], List[Tuple[str, Optional[str]]]],
        preprocessor: PhoBERTTextPreprocessor,
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Predict labels cho danh sách văn bản.
        - Nếu `texts` là `List[str]` -> input 1 câu.
        - Nếu `texts` là `List[Tuple[title, description]]` -> input 2 câu.
        
        Args:
            texts: Danh sách văn bản cần predict (1 câu hoặc cặp (title, description))
            preprocessor: PhoBERTTextPreprocessor để encode texts
            batch_size: Batch size cho prediction
            
        Returns:
            predictions: Mảng predicted labels
        """
        self.model.eval()
        all_predictions = []

        is_pair_input = len(texts) > 0 and isinstance(texts[0], tuple)
        if is_pair_input:
            titles = [t for (t, _) in texts]  # type: ignore[misc]
            descriptions = [d for (_, d) in texts]  # type: ignore[misc]
        else:
            titles = texts  # type: ignore[assignment]
            descriptions = None
        
        # Process in batches
        for i in range(0, len(titles), batch_size):
            batch_titles = titles[i:i + batch_size]
            batch_descriptions = descriptions[i:i + batch_size] if descriptions is not None else None
            
            # Encode texts (1 hoặc 2 câu)
            encoding = preprocessor.encode_texts(batch_titles, text_pairs=batch_descriptions)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
        
        return np.array(all_predictions)


class VietnameseNewsDataset(Dataset):
    """Custom Dataset cho Vietnamese News Classification.

    Input hiện hỗ trợ:
    - `title` (bắt buộc)
    - `description` (khuyến nghị; nếu không có sẽ fallback về title-only)
    - `label`
    """

    def __init__(
        self,
        csv_file: str,
        preprocessor: PhoBERTTextPreprocessor,
        label_encoder: Optional[LabelEncoder] = None,
        max_title_words: int = 20,
        max_description_words: int = 120,
        title_col: str = 'title',
        description_col: str = 'description',
        label_col: str = 'label',
    ):
        """Args:
        - csv_file: Path đến file CSV
        - preprocessor: PhoBERTTextPreprocessor để encode texts
        - label_encoder: LabelEncoder để encode labels (None sẽ tự tạo mới)
        - max_title_words: Giới hạn số từ của title trước khi tokenize
        - max_description_words: Giới hạn số từ của description trước khi tokenize
        - title_col/description_col/label_col: tên cột trong CSV
        """
        self.data = pd.read_csv(csv_file)
        self.preprocessor = preprocessor

        if title_col not in self.data.columns:
            raise ValueError(f"CSV must contain column '{title_col}'")
        if label_col not in self.data.columns:
            raise ValueError(f"CSV must contain column '{label_col}'")

        self.title_col = title_col
        self.description_col = description_col
        self.label_col = label_col

        # Encode labels
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.data[self.label_col])
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(self.data[self.label_col])

        self.titles = self.data[self.title_col].fillna('').astype(str).tolist()
        self.descriptions = (
            self.data[self.description_col].fillna('').astype(str).tolist()
            if self.description_col in self.data.columns
            else None
        )

        self.max_title_words = max_title_words
        self.max_description_words = max_description_words

    def __len__(self):
        return len(self.titles)

    @staticmethod
    def _truncate_words(text: str, max_words: int) -> str:
        if max_words <= 0:
            return ''
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text

    def __getitem__(self, idx):
        title = self._truncate_words(self.titles[idx], self.max_title_words)
        description = None
        if self.descriptions is not None:
            description = self._truncate_words(self.descriptions[idx], self.max_description_words)
            if description == '':
                description = None

        label = int(self.labels[idx])

        # Encode (title, description) nếu có; nếu không thì encode title-only
        encoding = self.preprocessor.encode_texts(
            [title],
            text_pairs=[description] if description is not None else None,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }